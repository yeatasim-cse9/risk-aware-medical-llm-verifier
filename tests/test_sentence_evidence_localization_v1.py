import sys
from pathlib import Path
import itertools
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from sentence_evidence_localization_v1 import sentences,localize,ranking_diagnostics

class Recipe:
    CWIDTH,COVERLAP,SWIDTH,SOVERLAP,BATCH=192,48,317,64,4
    @staticmethod
    def tiles(ids,width,overlap):
        out=[ids[:width]]
        while (len(out)-1)*(width-overlap)+len(out[-1])<len(ids):
            out.append(ids[len(out)*(width-overlap):len(out)*(width-overlap)+width])
        return out
class Engine:
    special=3
    def tokens(self,text): return list(range(len(text.split())))
    def encode(self,c,h): return (c,h)

def forward(engine,batch):
    return [dict(entailment=.7 if c[0]==0 else .2,
                 contradiction=.1 if c[0]==0 else .6,neutral=.2) for c,h in batch]

class TestLocalization(unittest.TestCase):
    def test_spans_abbreviation_decimal(self):
        text='Dr. A. measured 2.5 units. Next item!\n\nLast item'
        self.assertEqual([text[a:b] for a,b in sentences(text)],
                         ['Dr. A. measured 2.5 units.','Next item!','Last item'])
    def test_lossless_punctuation(self):
        text='  lower case? yes. "ok!"  end  '
        spans=sentences(text)
        self.assertEqual(''.join(text[a:b] for a,b in spans).replace(' ',''),
                         text.replace(' ',''))
    def test_localization_indices_and_pairs(self):
        r=localize(Engine(),'one. two.','word '*193,r9=Recipe,forward=forward,max_pairs=4)
        self.assertEqual(r['pairs'],4)
        self.assertEqual(r['localized_joint'],.6)
        self.assertEqual([u['support_window'] for u in r['units']],[0,0])
        self.assertEqual([u['conflict_window'] for u in r['units']],[1,1])
    def test_long_sentence_fully_tiled(self):
        r=localize(Engine(),'word '*318,'context',r9=Recipe,forward=forward,max_pairs=2)
        self.assertEqual(r['pairs'],2)
        self.assertEqual([u['segment'] for u in r['units']],[0,1])
    def test_budget_before_forward(self):
        def forbidden(*_): self.fail('forward called')
        with self.assertRaisesRegex(ValueError,'PAIR_BUDGET'):
            localize(Engine(),'one. two.','word '*193,r9=Recipe,forward=forbidden,max_pairs=3)
    def test_posterior_and_count(self):
        for f in (lambda *_:[],lambda *_:[dict(entailment=.8,contradiction=.8,neutral=0.)]):
            with self.assertRaises(ValueError):
                localize(Engine(),'one','context',r9=Recipe,forward=f,max_pairs=1)
    def test_ties_expected_aurc(self):
        rows=[('a',False,.1),('b',True,.1),('c',True,.2)]
        r=ranking_diagnostics(rows,coverages=(.25,.5,1.))
        expected=[]
        for yy in itertools.permutations((0,1)):
            z=list(yy)+[1];expected.append(sum(sum(z[:k])/k for k in range(1,4))/3)
        self.assertAlmostEqual(r['aurc'],sum(expected)/len(expected))
        self.assertEqual(r['coverage_points'][1]['accepted'],0)
        self.assertIsNone(r['coverage_points'][1]['risk'])
        self.assertEqual(r,ranking_diagnostics(rows[::-1],coverages=(.25,.5,1.)))
    def test_rank_invalid(self):
        for rows in ([],[('a',False,.1),('a',True,.2)],[('a',False,float('nan'))]):
            with self.assertRaises(ValueError):ranking_diagnostics(rows)
if __name__=='__main__':unittest.main()
