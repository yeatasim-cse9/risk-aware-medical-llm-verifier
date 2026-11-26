import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01f_r5_registry_v1 as r

def row(i):
    k=f"NCT{i:08d}"
    return {"id":k,"first_post":r.START,"links":["nct:"+k]}

class Registry(unittest.TestCase):
    def test_label_blind_selection(self):
        rows=[row(i) for i in range(5)]
        rows[1]["links"].append(rows[0]["links"][0])
        a,n=r.choose(rows,3)
        self.assertEqual(n,4)
        self.assertEqual((a,n),r.choose(list(reversed(rows)),3))
        self.assertFalse({rows[0]["id"],rows[1]["id"]}<=set(a))
    def test_id_projection(self):
        x={"protocolSection":{"identificationModule":{"nctId":"NCT00000001"},
             "statusModule":{"studyFirstPostDateStruct":{"date":r.START}}}}
        self.assertEqual(r.parse_id(x)["id"],"NCT00000001")
        x["protocolSection"]["statusModule"]["overallStatus"]="RECRUITING"
        with self.assertRaisesRegex(ValueError,"PROJECTION_SCHEMA"):
            r.parse_id(x)
    def test_fact_projection_and_membership(self):
        x={"protocolSection":{"identificationModule":{"nctId":"NCT00000001"},
             "statusModule":{"studyFirstPostDateStruct":{"date":r.START},"overallStatus":"COMPLETED"},
             "designModule":{"studyType":"INTERVENTIONAL","enrollmentInfo":{"count":20,"type":"ACTUAL"}}}}
        self.assertEqual(r.parse_fact(x,{"NCT00000001":r.START})["fields"]["enrollment"],{"count":20,"type":"ACTUAL"})
        with self.assertRaisesRegex(ValueError,"UNSELECTED_ID"):
            r.parse_fact(x,{})
        x["protocolSection"]["designModule"]["enrollmentInfo"]["type"]="ESTIMATED"
        self.assertEqual(r.parse_fact(x,{"NCT00000001":r.START})["fields"]["enrollment"]["type"],"ESTIMATED")
    def test_score(self):
        rid="NCT00000001"; ref={"count":20,"type":"ACTUAL"}
        self.assertEqual(r.score({"nct_id":rid,"value":ref},ref,rid),"correct")
        self.assertEqual(r.score({"nct_id":rid,"value":{"count":20,"type":"ESTIMATED"}},ref,rid),"wrong")
        self.assertEqual(r.score({"abstain":True},ref,rid),"abstain")
        self.assertEqual(r.score({"nct_id":rid,"value":{"count":True,"type":"ACTUAL"}},ref,rid),"invalid")
        self.assertEqual(r.score({"nct_id":"NCT00000002","value":ref},ref,rid),"wrong")
    def test_unauthorized_before_network(self):
        with patch.object(r,"get") as get:
            with self.assertRaisesRegex(ValueError,"EXPLICIT_RIGHTS"):
                r.prepare(Path("/nonexistent"),authorized=False)
            get.assert_not_called()
    def test_query_scope(self):
        with self.assertRaisesRegex(ValueError,"QUERY_SCOPE"):
            r.get("/studies",{})
        with self.assertRaisesRegex(ValueError,"FACT_QUERY_SCOPE"):
            r.get("/studies",{"format":"json","fields":r.FACT_FIELDS,"filter.ids":"NCT00000001","pageSize":100})
        with self.assertRaisesRegex(ValueError,"ENDPOINT"):
            r.get("/studies/NCT00000001")
    def test_selected_request(self):
        a=row(1)
        x={"protocolSection":{"identificationModule":{"nctId":a["id"]},
             "statusModule":{"studyFirstPostDateStruct":{"date":r.START}}}}
        with patch.object(r,"get",return_value={"studies":[x]}) as get:
            self.assertEqual(r.fetch_selected([a["id"]],[a,row(2)])[0]["id"],a["id"])
            self.assertEqual(get.call_args.args[1]["filter.ids"],a["id"])
    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError,"DUPLICATE_JSON_KEY"):
            r.strict_object([("a",1),("a",2)])

if __name__=="__main__":
    unittest.main()
