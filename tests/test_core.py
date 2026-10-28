import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
from scripts.synthetic import make_frame
from telco_churn.data import clean_frame, split_data, NUMERIC
from telco_churn.models import make_pipeline, make_search, fit_search, DT_GRID, SVM_GRID, choose_best
from telco_churn.feature_selection import feature_record
from telco_churn.evaluation import evaluate, score_model, plot_decision_tree

class DataTests(unittest.TestCase):
    def test_schema(self):
        with self.assertRaises(ValueError): clean_frame(make_frame().drop(columns="customerID"))
        with self.assertRaises(ValueError): clean_frame(make_frame().assign(extra=1))

    def test_target_mapping(self):
        df = make_frame(); X, y, _ = clean_frame(df)
        self.assertEqual(y.tolist(), df.Churn.map({"Yes": 1, "No": 0}).tolist())
        self.assertNotIn("customerID", X); self.assertNotIn("Churn", X)
        df.loc[0, "Churn"] = "unknown"
        with self.assertRaises(ValueError): clean_frame(df)

    def test_totalcharges_only_removal(self):
        df = make_frame(); df.loc[0, "TotalCharges"] = " "
        X, y, meta = clean_frame(df)
        self.assertEqual(meta["removed_rows"], 1); self.assertEqual(len(X), 159)
        self.assertEqual(X.TotalCharges.dtype.kind, "f")
        df.loc[1, "Partner"] = None
        with self.assertRaisesRegex(ValueError, "Partner"): clean_frame(df)

    def test_duplicate_id(self):
        df = make_frame(); df.loc[0, "customerID"] = df.loc[1, "customerID"]
        with self.assertRaisesRegex(ValueError, "unique"): clean_frame(df)

    def test_invalid_numeric(self):
        df = make_frame(); df.loc[0, "tenure"] = np.nan
        with self.assertRaises(ValueError): clean_frame(df)
        df = make_frame(); df.loc[0, "MonthlyCharges"] = np.inf
        with self.assertRaises(ValueError): clean_frame(df)

    def test_stratified_repeatable_split(self):
        X, y, _ = clean_frame(make_frame(200))
        a,b,c,d = split_data(X,y); e,f,g,h = split_data(X,y)
        self.assertEqual((len(a),len(b)), (140,60))
        self.assertFalse(set(a.index) & set(b.index))
        self.assertEqual(a.index.tolist(),e.index.tolist())
        self.assertLess(abs(c.mean()-d.mean()), .025)

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        cls.X, cls.y, _ = clean_frame(make_frame())
        cls.Xtr, cls.Xte, cls.ytr, cls.yte = split_data(cls.X,cls.y)

    @classmethod
    def tearDownClass(cls): cls.limits.restore_original_limits()

    def test_unfitted_and_grid_structure(self):
        for kind in ["decision_tree", "linear_svm"]:
            search = make_search(kind)
            self.assertEqual(list(search.estimator.named_steps), ["preprocess", "rfe", "classifier"])
            self.assertFalse(hasattr(search.estimator.named_steps["preprocess"], "transformers_"))
            self.assertFalse(hasattr(search.estimator.named_steps["rfe"], "support_"))
            self.assertEqual(search.cv.n_splits, 5); self.assertTrue(search.cv.shuffle)
        self.assertEqual(len(ParameterGrid(DT_GRID)),72)
        self.assertEqual(len(ParameterGrid(SVM_GRID)),9)
        self.assertEqual(choose_best({"mean_test_ap":[.7,.7,.6]}),0)

    def test_unknown_categories_and_training_scaler(self):
        pipe = make_pipeline("rbf_svm").fit(self.Xtr,self.ytr)
        prep = pipe.named_steps["preprocess"]
        numeric = prep.named_transformers_["numeric"]
        self.assertIsInstance(numeric, StandardScaler)
        np.testing.assert_allclose(numeric.mean_, self.Xtr[NUMERIC].mean())
        new = self.Xte.copy(); new["Contract"] = "unseen"
        before = numeric.mean_.copy()
        self.assertTrue(np.isfinite(prep.transform(new)).all())
        np.testing.assert_array_equal(before,numeric.mean_)

    def test_exact_feature_names_and_input_order(self):
        for kind in ["decision_tree","linear_svm","rbf_svm"]:
            with self.subTest(kind=kind):
                pipe = make_pipeline(kind).fit(self.Xtr,self.ytr)
                record = feature_record(pipe)
                self.assertEqual(record["selected_feature_count"],20)
                transformed = pipe.named_steps["preprocess"].transform(self.Xte)
                support = pipe.named_steps["rfe"].support_
                np.testing.assert_allclose(pipe[:-1].transform(self.Xte),transformed[:,support])
                names = pipe.named_steps["preprocess"].get_feature_names_out()[support].tolist()
                self.assertEqual(record["selected_feature_names"],names)
                self.assertEqual(pipe.predict(self.Xte).shape,(len(self.Xte),))
                self.assertIsInstance(pipe.named_steps["preprocess"].named_transformers_["numeric"],StandardScaler)

    def test_fold_fit_inputs_and_labels(self):
        prep_calls, rfe_calls, scaler_calls = [], [], []
        prep_fit = ColumnTransformer.fit_transform; rfe_fit = RFE.fit; scaler_fit = StandardScaler.fit
        def prep_spy(obj,X,y=None,**kwargs):
            prep_calls.append((list(X.index), np.asarray(y).copy()))
            return prep_fit(obj,X,y,**kwargs)
        def rfe_spy(obj,X,y,**kwargs):
            rfe_calls.append((len(X), np.asarray(y).copy()))
            return rfe_fit(obj,X,y,**kwargs)
        def scaler_spy(obj,X,y=None,**kwargs):
            scaler_calls.append(list(X.index))
            return scaler_fit(obj,X,y,**kwargs)
        with patch.object(ColumnTransformer,"fit_transform",prep_spy), patch.object(RFE,"fit",rfe_spy), patch.object(StandardScaler,"fit",scaler_spy):
            search = fit_search(self.Xtr,self.ytr,"decision_tree",grid={"classifier__max_depth":[3]})
        expected = [(list(self.Xtr.iloc[a].index),self.ytr.iloc[a].to_numpy()) for a,b in search.cv.split(self.Xtr,self.ytr)]
        expected.append((list(self.Xtr.index),self.ytr.to_numpy()))
        self.assertEqual(len(prep_calls),6); self.assertEqual(len(rfe_calls),6)
        self.assertEqual(len(scaler_calls),6)
        for i, (ids,labels) in enumerate(expected):
            self.assertEqual(prep_calls[i][0],ids); self.assertEqual(scaler_calls[i],ids)
            self.assertFalse(set(ids)&set(self.Xte.index))
            np.testing.assert_array_equal(prep_calls[i][1],labels)
            self.assertEqual(rfe_calls[i][0],len(ids))
            np.testing.assert_array_equal(rfe_calls[i][1],labels)

    def test_changed_holdout_labels_cannot_change_selection(self):
        self.assertNotIn("X_test",inspect.signature(fit_search).parameters)
        a = fit_search(self.Xtr,self.ytr,"decision_tree",grid={"classifier__max_depth":[2,3]})
        flipped = 1-self.yte
        evaluate(flipped,a.predict(self.Xte),score_model(a,self.Xte)[0])
        b = fit_search(self.Xtr,self.ytr,"decision_tree",grid={"classifier__max_depth":[2,3]})
        self.assertEqual(a.best_params_,b.best_params_)
        self.assertEqual(feature_record(a.best_estimator_),feature_record(b.best_estimator_))

    def test_plot_uses_best_fitted_classifier(self):
        search = fit_search(self.Xtr,self.ytr,"decision_tree",grid={"classifier__max_depth":[2,3]})
        from sklearn.tree import plot_tree as original
        with tempfile.TemporaryDirectory() as tmp, patch("telco_churn.evaluation.plot_tree", wraps=original) as spy:
            path = Path(tmp)/"tree.png"; plot_decision_tree(search.best_estimator_,path)
            self.assertIs(spy.call_args.args[0],search.best_estimator_.named_steps["classifier"])
            self.assertEqual(len(spy.call_args.kwargs["feature_names"]),20)
            self.assertGreater(path.stat().st_size,1000)

    def test_baseline_feature_setting(self):
        for kind in ["dummy","logistic"]:
            pipe = make_pipeline(kind).fit(self.Xtr,self.ytr)
            self.assertEqual(feature_record(pipe)["feature_setting"],"all_transformed")
            self.assertGreater(feature_record(pipe)["selected_feature_count"],20)

    def test_svm_continuous_scores(self):
        pipe=make_pipeline("linear_svm").fit(self.Xtr,self.ytr)
        score,kind=score_model(pipe,self.Xte)
        np.testing.assert_array_equal(score,pipe.decision_function(self.Xte))
        self.assertIn("not calibrated",kind)

class MetricTests(unittest.TestCase):
    def test_known_metrics(self):
        m=evaluate([0,0,1,1],[0,1,0,1],[.1,.8,.3,.9])
        for key in ["accuracy","precision","recall","f1"]: self.assertEqual(m[key],.5)
        self.assertEqual(m["roc_auc"],.75)
        self.assertAlmostEqual(m["average_precision"],5/6)
        self.assertEqual(m["confusion_matrix"],[[1,1],[1,1]])

    def test_no_positive_predictions(self):
        m=evaluate([0,0,1,1],[0,0,0,0],[0,0,0,0])
        self.assertEqual(m["precision"],0);self.assertEqual(m["recall"],0);self.assertEqual(m["f1"],0)
        self.assertEqual(m["average_precision"],.5);self.assertEqual(m["roc_auc"],.5)

if __name__ == "__main__": unittest.main()
