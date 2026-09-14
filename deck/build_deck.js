const P=require("pptxgenjs");
const pres=new P(); pres.defineLayout({name:"W",width:13.333,height:7.5}); pres.layout="W";
const PAPER="FDFDFC",INK="1B1A17",MUTE="6B6963",RULE="1B1A17",HAIR="C9C6BD",TINT="FDFDFC",
      ACC="6B6963",GRN="1B1A17",FLAG="8A4A2C";
const HEAD="Georgia",BODY="Georgia";
const W=13.333,H=7.5,M=0.85,CW=W-2*M;
function slide(k,t,s){const x=pres.addSlide();x.background={color:PAPER};
 x.addText(k,{x:M,y:0.40,w:CW,h:0.22,fontFace:BODY,fontSize:10.5,color:MUTE,charSpacing:0.7,margin:0});
 x.addText(t,{x:M,y:0.66,w:CW,h:0.54,fontFace:HEAD,fontSize:25,color:INK,margin:0});
 if(s)x.addText(s,{x:M,y:1.24,w:CW,h:0.30,fontFace:BODY,fontSize:12,color:MUTE,italics:true,margin:0});
 x.addShape(pres.ShapeType.rect,{x:M,y:s?1.60:1.32,w:CW,h:0.008,fill:{color:RULE}});return x;}
function tint(s,x,y,w,h,c){s.addShape(pres.ShapeType.rect,{x,y,w:w,h:0.006,fill:{color:HAIR}});}
function lab(s,x,y,w,t,o){o=o||{};s.addText(t,{x,y,w,h:0.22,fontFace:BODY,fontSize:o.size||10.5,italics:true,color:o.color||MUTE,charSpacing:0.3,margin:0});}
function tbl(s,rows,opt){opt=opt||{};const ctr=opt.centerCols||[],hi=opt.hiRows||[];
 const sh=rows.map((r,ri)=>r.map((c,ci)=>{const o={};
  if(ri===0){o.bold=true;o.border=[{type:"solid",color:RULE,pt:1.1},{type:"none"},
                                   {type:"solid",color:RULE,pt:0.6},{type:"none"}];}
  else {o.border=[{type:"none"},{type:"none"},
                  {type:"solid",color:ri===rows.length-1?RULE:"FFFFFF",pt:ri===rows.length-1?1.1:0},{type:"none"}];}
  if(hi.indexOf(ri)!==-1)o.bold=true;
  if(ctr.indexOf(ci)!==-1)o.align="center";return {text:String(c),options:o};}));
 const t=Object.assign({x:M,w:CW,fontFace:BODY,
  fontSize:10,color:INK,valign:"top",margin:4},opt);
 delete t.centerCols;delete t.hiRows;s.addTable(sh,t);}

/* 1 TITLE */
{const s=pres.addSlide();s.background={color:PAPER};
 s.addText("Foundations of Data Science  ·  Review 1",{x:M,y:0.62,w:CW,h:0.26,fontFace:BODY,fontSize:11.5,color:MUTE,charSpacing:0.6,margin:0});
 s.addText("Instruction-Guided Edit-Region Prediction",{x:M,y:0.96,w:CW,h:0.62,fontFace:HEAD,fontSize:33,color:INK,margin:0});
 s.addText("Predicting where an instruction says to edit, at 1% of the parameters",{x:M,y:2.16,w:CW,h:0.36,fontFace:BODY,fontSize:14,color:MUTE,italics:true,margin:0});
 s.addShape(pres.ShapeType.rect,{x:M,y:2.70,w:CW,h:0.008,fill:{color:RULE}});
 s.addText([{text:"Amritha S",options:{bold:true}},{text:"   23BEC1368"}],{x:M,y:2.96,w:5.6,h:0.3,fontFace:BODY,fontSize:14,color:INK,margin:0});
 s.addText([{text:"Yugeshwaran P",options:{bold:true}},{text:"   23BEC1404"}],{x:M,y:3.32,w:5.6,h:0.3,fontFace:BODY,fontSize:14,color:INK,margin:0});
 s.addText("Guide  ·  Dr. Saranya M  (54783)\nSchool of Electronics Engineering  ·  VIT Chennai",{x:M,y:3.88,w:6.2,h:0.7,fontFace:BODY,fontSize:12,color:MUTE,lineSpacing:17,margin:0});
 lab(s,W-M-4.55,3.02,4.3,"Headline result",{color:MUTE});
 s.addText([{text:"IoU 0.2065",options:{bold:true}},{text:"  ·  beats CLIPSeg (0.1849) and the human annotations (0.1511)\n"},
            {text:"+120%",options:{bold:true}},{text:"  on insertion vs CLIPSeg\n"},
            {text:"10.3×",options:{bold:true}},{text:"  faster, "},{text:"32×",options:{bold:true}},{text:"  fewer trainable parameters"}],
   {x:W-M-4.55,y:3.30,w:4.3,h:1.4,fontFace:BODY,fontSize:11.5,color:INK,lineSpacing:17,margin:0});
 s.addShape(pres.ShapeType.rect,{x:0,y:H-0.5,w:W,h:0.012,fill:{color:RULE}});
 s.addNotes("Open with the one-sentence framing, not the architecture.\n\n\"Instruction-guided editors are very good at editing. They are silent about WHERE to edit. We built the thing that decides where, and measured it — which nobody does.\"\n\nHeadline, say it plainly: IoU 0.2065 over 12 seeds on 503 held-out images. That beats CLIPSeg, a 150-million-parameter referring segmenter, and it beats the human annotations MagicBrush ships. We use 4.7 million trainable parameters and run 10x faster per instruction.\n\nEverything trained on this laptop. About 21 hours of compute. No cloud.\n\nDON'T oversell here. The insertion ceiling slide later is where we show we know the limits — that is what makes the rest credible.");}

/* 2 PROBLEM */
{const s=slide("The problem","An editor must decide where to edit before it can edit","Everyone works on the drawing. Almost nobody measures the decision.");
 const r=[["Instruction","What must change","An existing object?"],
  ["make the jacket blue","the jacket, not the person","No — a part"],
  ["remove the cat on the left","one instance, by position","Yes, but which?"],
  ["put a hat on the man","empty space above his head","No — nothing is there"]];
 tbl(s,r,{y:1.88,colW:[3.6,4.6,3.43],rowH:0.48,fontSize:11.5});
 tint(s,M,4.15,CW,0.92);
 s.addText("Referring segmentation returns what a phrase denotes. For “put a hat on the man” the phrase denotes the man — the region that must change is above his head.",
  {x:M+0.32,y:4.34,w:CW-0.64,h:0.58,fontFace:BODY,fontSize:11.5,color:INK,lineSpacing:16,margin:0});
 lab(s,M,5.30,9,"Why this is a data-science problem");
 s.addText("No benchmark scores a predicted edit region. We checked the base paper directly: the word “IoU” appears once, qualitatively. A 2026 paper whose headline contribution is a MaskPredictor never measures how well it localizes.",
  {x:M,y:5.58,w:CW,h:0.7,fontFace:BODY,fontSize:11.5,color:INK,lineSpacing:16,margin:0});
 s.addNotes("The table is the argument. Walk one row: \"make the jacket blue\" — the region is a PART of an object. \"put a hat on the man\" — the region is empty space above his head. Nothing is there to segment.\n\nThat last row is the whole project. Referring segmentation returns what a phrase DENOTES. For \"put a hat on the man\" the phrase denotes the man. The correct answer is the air above him.\n\nThen the data-science point at the bottom: we checked the base paper directly. A 2026 paper whose headline contribution is a MaskPredictor never reports a mask IoU. Not once, quantitatively.\n\nIf asked why: there is no benchmark that would have caught it. We show that in the survey — the closest benchmark, LocateEdit-Bench, scores edits that already happened.");}

/* 3 BASE PAPER */
{const s=slide("Base paper","AdaptEdit — Edit Where You Mean  (arXiv:2604.23763, April 2026)","The only 2026 work that trains a predictor to ground the edit region from the instruction.");
 const a=[["§","Component","Params"],
  ["3.1","Qwen-Image-Edit DiT, 60 blocks — frozen","20.43 B"],
  ["3.2","Block Adapters ×60","4.67 B"],
  ["3.3","Condition Encoder","351 M"],
  ["3.4","SpatialGate","—"],
  ["3.6","MaskPredictor — FiLM query-grid decoder","7.8 M"]];
 tbl(s,a,{y:1.88,w:6.4,colW:[0.5,4.3,1.6],rowH:0.34,fontSize:10,hiRows:[5]});
 tint(s,M+6.8,1.88,CW-6.8,2.25);
 lab(s,M+7.05,2.08,4.3,"Its own stated limitation",{color:FLAG});
 s.addText("“Geometry-changing edits with no localized source region (‘put a hat on the dog’ where the dog has no hat): the GT mask covers the new content’s location but there is no clean source signal to ‘protect’ outside it.”",
  {x:M+7.05,y:2.38,w:4.3,h:1.65,fontFace:HEAD,fontSize:11,italics:true,color:FLAG,lineSpacing:15,margin:0});
 lab(s,M,4.05,10,"What we can and cannot do with it");
 const b=[["Cannot","run or retrain it","20.43 B frozen + 5.0 B trainable needs multi-GPU"],
  ["Cannot","reproduce its numbers","it reports L1 / CLIP-I / DINO on edited images, never a mask IoU"],
  ["Can","adopt its formulation","predict the region from the instruction; compare on its data"],
  ["Can","reimplement its mechanism","the MaskPredictor head, at our scale, under matched conditions"]];
 tbl(s,b,{y:4.33,colW:[1.0,2.5,8.13],rowH:0.34,fontSize:10});
 s.addNotes("Keep this short — it is context, not contribution.\n\nAdaptEdit, April 2026. Freezes a 20-billion-parameter diffusion transformer and injects adapters. The part we care about is one thin head: the MaskPredictor, which grounds the edit region from the instruction.\n\nTwo facts matter. One: we cannot reproduce it — 20.43 billion frozen plus about 5 billion trainable. Not on a 26 GB laptop, not on anything we have.\n\nTwo, and this is the opening: its MaskPredictor is supervised on MagicBrush human annotation and rendered text boxes. Explicitly NOT pixel differences. Read the limitation quote off the slide — they name our gap themselves.\n\nIf asked \"why not just use their numbers\": they report no mask IoU. There is nothing to compare to. That absence IS the contribution.");}

/* 4 SUPERVISION */
{const s=slide("Finding 1","The supervision the field trains on is 9.1× too coarse","All 528 MagicBrush dev turns, against the region that actually changed.");
 const r=[["edit type","n","human mask","actually changed","ratio","recall","precision","IoU"],
  ["insert","103","60.0%","7.0%","11.9×","97.8%","11.9%","0.12"],
  ["modify","390","62.9%","11.2%","8.5×","96.5%","17.1%","0.17"],
  ["remove","35","63.6%","8.9%","7.1×","96.2%","13.4%","0.13"],
  ["all","528","62.4%","10.3%","9.1×","96.7%","15.8%","0.16"]];
 tbl(s,r,{y:1.88,colW:[1.5,0.8,1.75,2.0,1.1,1.3,1.35,1.83],rowH:0.38,fontSize:10.5,centerCols:[1,2,3,4,5,6,7],hiRows:[4]});
 tint(s,M,4.10,CW,1.05);
 s.addText("Recall 96.7% — the annotations are correct.  Precision 15.8% — only a sixth of the annotated region actually changes. These are region-of-interest scribbles for an inpainting tool, not edit regions.",
  {x:M+0.32,y:4.32,w:CW-0.64,h:0.7,fontFace:BODY,fontSize:12,color:INK,lineSpacing:17,margin:0});
 lab(s,M,5.35,9,"So the supervision is the opportunity");
 s.addText("We keep MagicBrush’s real human before/after pairs but derive the target ourselves by CIE76 ΔE in L*a*b*. Where an object actually landed is a far tighter signal than where a human roughly brushed.",
  {x:M,y:5.63,w:CW,h:0.65,fontFace:BODY,fontSize:11.5,color:INK,lineSpacing:16,margin:0});
 s.addNotes("This is the measurement that justifies our supervision choice.\n\nWe measured MagicBrush's own masks across all 528 dev turns. They cover 62.4 percent of the frame. The real changed area is 10.3 percent. That is 9.1 times too large. Precision 15.8 percent. Worst on insertions — 11.9 times.\n\nThey are region-of-interest scribbles. And they are what the field trains on, including our base paper.\n\nSo we do not use them as targets. We derive the target ourselves by CIE76 delta-E in L*a*b* between the real source and target images. Where a hat LANDED is ground truth for where a hat SHOULD GO.\n\nMention the polarity bug if it comes up naturally: MagicBrush masks are bright-means-PRESERVE. We assumed the opposite once and got IoU 0.001. We caught it, fixed it, recomputed everything.");}

/* 5 ARCHITECTURE */
{const s=slide("Our architecture","One mechanism changed, and it is significant","Frozen 71.75 M backbone + frozen CLIP + a 1.4 M trainable head.");
 lab(s,M,1.88,6,"Theirs — decode from learned queries");
 s.addText("M = σ( Upsample( ConvHead( CA(q = γ⊙Q + β,  k,v = φ(features)) ) ) )",
  {x:M,y:2.14,w:6.0,h:0.32,fontFace:"Consolas",fontSize:10.5,color:INK,margin:0});
 s.addText("A learned 16×16 query grid cross-attends to patchified features. Spatial structure is learned from scratch.",
  {x:M,y:2.50,w:6.0,h:0.5,fontFace:BODY,fontSize:11,color:MUTE,lineSpacing:15,margin:0});
 lab(s,M,3.12,6,"Ours — a coefficient field over an existing basis",{color:GRN});
 s.addText("M = σ( Σ cᵢ(x,y) · protoᵢ(x,y) + b )",{x:M,y:3.38,w:6.0,h:0.34,fontFace:"Consolas",fontSize:12.5,bold:true,color:GRN,margin:0});
 s.addText("FastSAM already emits 32 prototype masks. Their coefficients normally come from a detection — which is why its output can only be an existing object. The instruction predicts them instead, as a 20×20 spatial field, so different regions use different mixtures.",
  {x:M,y:3.76,w:6.0,h:1.0,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 tint(s,M+6.5,1.88,CW-6.5,2.35);
 lab(s,M+6.75,2.08,4.5,"Measured head-to-head, matched conditions",{color:GRN});
 const h=[["head","params","IoU"],
  ["their mechanism","923 K","0.1050"],
  ["ours, global coeff","1.4 M","0.1907"],
  ["ours, spatial field","4.7 M","0.2065"]];
 tbl(s,h,{x:M+6.75,y:2.38,w:4.4,colW:[2.0,1.1,1.3],rowH:0.32,fontSize:10,centerCols:[1,2],hiRows:[3]});
 s.addText("Identical inputs, data, loss, schedule and seeds; 8,306 samples.\nSpatial vs global:  p = 5.4e-06,  Cohen's d = 2.50  (12 seeds).\nTheir mechanism was measured at 2,278 only.",
  {x:M+6.75,y:3.74,w:4.4,h:0.62,fontFace:BODY,fontSize:9.5,color:INK,lineSpacing:13,margin:0});
 tint(s,M+6.5,4.45,CW-6.5,1.15);
 lab(s,M+6.75,4.62,4.5,"What is shared, not claimed",{color:FLAG});
 s.addText("FiLM modulation and BCE + Dice are their design too — we reached the same choices independently.",
  {x:M+6.75,y:4.88,w:4.4,h:0.6,fontFace:BODY,fontSize:10.5,color:FLAG,lineSpacing:14,margin:0});
 s.addNotes("Two heads, and the difference is structural, not capacity.\n\nv1 predicts ONE coefficient vector per image. The mask is a single linear combination of the 32 prototypes, applied uniformly everywhere.\n\nv2 predicts a 20-by-20 FIELD of coefficients. Different locations use different prototype mixtures. That is the hypothesis: a spatially varying combination should help where the region is NOT the extent of one object — which is exactly insertion.\n\nThe table: v2 0.2065 against v1 0.1907. p equals 5.4 times ten to the minus six, Cohen's d of 2.50. Large effect, not just significant.\n\nBe honest about what is shared: FiLM modulation and BCE-plus-Dice are AdaptEdit's choices too. We reached them independently — that is not a novelty claim and we do not make one.");}

/* 6 RESULTS FIGURE */
{const s=slide("Results","Every seed, at both training-set sizes","503 held-out dev samples · 8,306 training samples · 12 seeds per configuration");
 s.addImage({path:"img/results.png",x:M,y:1.80,w:7.05,h:7.05/1.739});
 const mt=[["metric","v2 spatial","v1 global"],
  ["IoU","0.2065","0.1907"],
  ["Precision","0.3126","0.2972"],
  ["Recall","0.3878","0.3510"],
  ["F1 / Dice","0.2976","0.2757"]];
 tbl(s,mt,{x:M+7.35,y:1.92,w:4.28,colW:[1.68,1.34,1.26],rowH:0.34,fontSize:10,centerCols:[1,2],hiRows:[1]});
 tint(s,M+7.35,4.00,4.28,1.62);
 lab(s,M+7.55,4.18,3.9,"Against the baselines",{color:GRN});
 s.addText("CLIPSeg  0.1849   +0.0216  p=5.2e-07\nHuman masks  0.1511   +0.0554\nAll 12 seeds clear CLIPSeg.",
  {x:M+7.55,y:4.44,w:3.9,h:1.1,fontFace:BODY,fontSize:10.5,color:INK,lineSpacing:15,margin:0});
 tint(s,M,6.28,CW,0.80);
 s.addText([{text:"0.2065 ± 0.0072",options:{bold:true}},{text:"  —  now above CLIPSeg (0.1849) as well as the human masks, at "},
  {text:"32× fewer trainable parameters",options:{bold:true}},{text:" and "},{text:"10.3× lower per-instruction latency",options:{bold:true}},
  {text:".  Insertion 0.1186 vs CLIPSeg 0.054 — +120%."}],
  {x:M+0.32,y:6.44,w:CW-0.64,h:0.56,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 s.addNotes("This is the results slide. Take it slowly.\n\nEvery dot is one seed. Four groups: both heads at 2,278 samples and at 8,306. The point of plotting every seed is that the separation is not carried by an outlier — all 12 v2 seeds at 8,306 sit above the CLIPSeg line.\n\nNumbers: 0.2065 plus or minus 0.0072. Against CLIPSeg 0.1849, p equals 5.2 times ten to the minus seven. Against human masks 0.1511.\n\nThe side table is the full metric suite — precision 0.3126, recall 0.3878, F1 0.2976 — because IoU alone hides the trade-off.\n\nSAY THIS OUT LOUD, do not let them find it: the win is data first, architecture second. Both heads gained about the same from more data, and v1 at 8,306 ALSO clears CLIPSeg. The spatial field adds a real, significant 0.0158 on top. But the bigger single factor was the training set.");}

/* 6b KEY PARAMETERS */
{const s=slide("Configuration","Every parameter behind those numbers","Taken from the source, not from notes");
 const c=[["group","parameter","value"],
  ["Supervision","ΔE metric / threshold","CIE76 in L*a*b*  ·  12.0"],
  ["","morph kernel · min component","5×5  ·  40 px"],
  ["","sample filter (changed area)","0.4% – 45%"],
  ["Representation","input · prototypes","640×640  ·  32 at 160×160"],
  ["","text · image context","512-d CLIP  ·  640-d SPPF   (both frozen)"],
  ["Head (v2)","coefficient grid · hidden","20×20 bilinear  ·  512  ·  dropout 0"],
  ["Optimisation","AdamW · lr · decay","1e-3 cosine to 0  ·  0.01"],
  ["","batch · loss · clip","32  ·  BCE + 2×Dice  ·  norm 1.0"],
  ["Evaluation","epochs · selection","60  ·  best on train-val, never dev"],
  ["","threshold · seeds","0.5 fixed  ·  12 (1368, 1–11)"],
  ["","baseline","CIDAS/clipseg-rd64-refined"]];
 tbl(s,c,{y:1.78,colW:[2.0,3.1,6.53],rowH:0.355,fontSize:10});
 tint(s,M,6.32,CW,0.78);
 s.addText("8,306 usable training turns from all 51 MagicBrush shards (8,807 rows, 501 rejected as degenerate).  Splits verified image-disjoint: 794 dev images, 13,317 train, 0 overlap.",
  {x:M+0.32,y:6.50,w:CW-0.64,h:0.5,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 s.addNotes("This slide exists so nobody has to ask \"what were your settings\".\n\nDo not read it. Say: \"every value here came out of the source, not out of our notes — delta-E threshold 12, chosen from a noise-robustness sweep; 32 frozen prototypes at 160 by 160; AdamW at 1e-3; BCE plus 2x Dice; 12 seeds, everything seeded including MPS.\"\n\nThen point at the bottom line, which is the one that matters: 8,306 usable turns from all 51 shards, and the splits are image-disjoint — 794 dev images, 13,317 train, zero overlap.\n\nIf asked why the leakage check: MagicBrush is multi-turn, so one photo produces several edits. If the split were by turn instead of by image, scaling the training set 3.65 times would have quietly contaminated everything. We hashed every image on both sides BEFORE trusting any result.");}

/* 6c OPERATING POINT */
{const s=slide("Operating point","IoU alone hides the precision/recall trade-off","Threshold chosen on train-val, never on dev");
 const t1=[["threshold","IoU","precision","recall","predicted area"],
  ["0.2","0.2172","0.2637","0.5775","22.74%"],
  ["0.3","0.2180","0.2818","0.5063","18.10%"],
  ["0.5  (reported)","0.2065","0.3126","0.3878","11.70%"],
  ["0.7","0.1747","0.3458","0.2750","6.90%"],
  ["0.9","0.1119","0.3705","0.1479","2.74%"]];
 tbl(s,t1,{y:1.80,colW:[2.5,2.2,2.3,2.3,2.33],rowH:0.38,fontSize:10.5,centerCols:[1,2,3,4],hiRows:[3]});
 s.addText("True changed area is 9.58%, so at 0.5 the model over-predicts slightly.",
  {x:M,y:4.28,w:CW,h:0.3,fontFace:BODY,fontSize:10.5,color:MUTE,italics:true,margin:0});
 tint(s,M,4.66,CW,1.32);
 lab(s,M+0.32,4.84,8,"Selected honestly, on the train-val slice",{color:GRN});
 s.addText("v2 picks 0.3 (×7) or 0.2 (×5) → dev 0.2185 ± 0.0051, +0.0120 over fixed 0.5 (paired p = 2.9e-07). Insertion rises to 0.134, 63% of its ceiling.",
  {x:M+0.32,y:5.12,w:CW-0.64,h:0.76,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 tint(s,M,6.10,CW,0.98);
 lab(s,M+0.32,6.26,8,"Why the headline still quotes 0.5",{color:FLAG});
 s.addText("CLIPSeg is evaluated at its default threshold, so a tuned comparison would tilt toward us. We report the conservative number. With both heads tuned, the v2 lead narrows from +0.0158 to +0.0093.",
  {x:M+0.32,y:6.52,w:CW-0.64,h:0.5,fontFace:BODY,fontSize:10.5,color:FLAG,lineSpacing:14,margin:0});
 s.addNotes("This is the slide that shows methodological discipline. Lead with the honest version.\n\n\"We fixed the threshold at 0.5 before evaluating. Then we swept it, and dev actually peaks at 0.3.\"\n\nNow the important part: adopting 0.3 because DEV says so would be test-set tuning. Earlier in this project, selecting the best epoch on dev inverted our v1-versus-v2 conclusion entirely. So we selected the threshold per seed on the train-val slice instead, and touched dev once. That gives 0.2185.\n\nTwo things we volunteer rather than hide:\n\nOne — with both heads tuned, the v2 lead narrows from 0.0158 to 0.0093. Part of its edge at 0.5 was that 0.5 happened to suit it.\n\nTwo — CLIPSeg is at its default operating point, so a tuned comparison tilts toward us. That is why the headline still quotes 0.5, the conservative number.");}

/* 7 ANALYSIS FIGURE */
{const s=slide("Training dynamics","The best epoch is 2–7 of 60, and language carries the signal","3 seeds per head at 8,306 samples · ▼ marks the epoch actually selected");
 s.addImage({path:"img/curves.png",x:M,y:1.74,w:7.9,h:7.9/1.778});
 const r=[["Overfits fast","validation IoU peaks at epoch 2–7 then halves by epoch 60, while training loss falls throughout"],
  ["Selection saved it","choosing on a held-out train slice is why the reported numbers hold; the last epoch would have reported half"],
  ["Language is real","zeroing the instruction collapses IoU from 0.188 to ~0.03 — a gap of +0.158"],
  ["Reproducible","all 6 curve seeds reproduced their 12-seed dev IoU exactly, to four decimals"],
  ["Headroom is bounded","least squares caps ANY predictor over this basis at 0.2119 insert / 0.3867 modify / 0.4325 remove — we reach 56% / 60% / 46%"]];
 tbl(s,r,{x:M+8.15,y:1.86,w:3.48,colW:[1.2,2.28],rowH:0.74,fontSize:8.5});
 tint(s,M,6.22,CW,0.85);
 s.addText("Latency, resolution-matched at CLIPSeg's native 352 px:  v2 per-instruction 5.30 ms vs 54.3 ms = 10.3×, at 32× fewer trainable parameters.  The backbone runs once per image (33.6 ms); each further instruction costs only the head.",
  {x:M+0.32,y:6.40,w:CW-0.64,h:0.56,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 s.addNotes("Two findings here.\n\nFirst, overfitting. The best epoch is between 2 and 7 out of 60, every time. Validation IoU peaks almost immediately and then halves by epoch 60, while training loss keeps falling the whole way. Selecting on a held-out slice of train is the only reason our numbers hold — training to the last epoch would have reported about half the IoU.\n\nPractical note worth saying: 60 epochs was about six times more than needed. Any further experiment on this pipeline is cheap now.\n\nSecond, the language ablation. Zero the instruction and validation IoU collapses from 0.188 to about 0.03. The head is reading the instruction, not exploiting an image prior.\n\nBottom line is latency: 5.30 ms per instruction against CLIPSeg's 54.3, resolution-matched. The backbone runs once per image; each extra instruction costs only the head. That split is why it amortises.");}

/* 8 STAGE 2 */
{const s=slide("Stage 2","Editing inside the predicted region — the premise, tested","60 held-out dev samples · one editor (InstructPix2Pix), only the gating mask varies");
 const r=[["gating mask","recall of the edit ↑","collateral change ↓","net ↑"],
  ["none — whole frame","33.3%","24.2%","+9.1%"],
  ["ours, as predicted","7.5%","1.5%","+6.0%"],
  ["ours, tuned — peak of the sweep","22.9%","8.8%","+14.2%"],
  ["MagicBrush human mask","31.5%","13.0%","+18.5%"],
  ["ground-truth region","30.2%","0.1%","+30.1%"]];
 // second arm: a real inpainting model, where the mask decides what is regenerated
 tbl(s,r,{y:1.88,colW:[4.2,2.6,2.6,2.23],rowH:0.40,fontSize:10.5,centerCols:[1,2,3],hiRows:[3]});
 tint(s,M,4.50,6.1,1.55);
 lab(s,M+0.3,4.70,5.5,"The premise holds",{color:GRN});
 s.addText("Editing inside the correct region is worth 3.3× — net +30.1% against +9.1% for editing the whole frame. Our tuned mask reaches +14.2%, 1.56× better than the status quo.",
  {x:M+0.3,y:4.98,w:5.5,h:0.9,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 tint(s,M+6.4,4.50,CW-6.4,1.55);
 lab(s,M+6.7,4.70,4.5,"And the diagnosis is precise");
 s.addText("Our mask is precision-biased — 1.5% collateral but only 7.5% recall, so gating discarded the edit. Dilation trades precision for recall; swept to 200 px, net peaks at 64 px and falls away either side.",
  {x:M+6.7,y:4.98,w:4.3,h:0.9,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 s.addText("Coarse supervision costs at the editing stage too, not only on the mask metric: the MagicBrush mask — measured 9.1× too large — loses 11.6 points of net against the ground-truth region.",
  {x:M,y:6.22,w:CW,h:0.5,fontFace:BODY,fontSize:10.5,color:MUTE,italics:true,lineSpacing:14,margin:0});
 s.addNotes("This closes the loop and tests the premise nobody had tested: does predicting the region actually improve the EDIT?\n\nSame editor for every row — InstructPix2Pix. Only the gating mask changes. So any difference is the mask.\n\nThe headline: editing inside the correct region is worth 3.3 times. Plus 30.1 percent net for the ground-truth region against plus 9.1 for whole-frame editing.\n\nOur mask as predicted was too conservative — 1.5 percent collateral but only 7.5 percent recall, so gating threw away most of the edit. Dilating to 64 pixels fixes it: plus 14.2 percent. We swept to 200 pixels and confirmed 64 is a true peak, not the edge of the search.\n\nBE HONEST: we do not beat the MagicBrush human mask here — 14.2 against 18.5. Their masks are far too large but retain more of the edit. Closing that is the same representational problem seen from the other end.");}

/* 8b STAGE 2 WITH REAL INPAINTING */
{const s=slide("Stage 2, part two","With a real inpainting model, localization is worth 5.8×","60 held-out dev samples · the mask now decides what gets regenerated, not just what is kept");
 const r=[["gating mask","recall ↑","collateral ↓","net ↑"],
  ["none — whole frame","33.3%","24.2%","+9.1%"],
  ["ours, tuned (thr 0.3, 32 px)","61.7%","22.2%","+39.5%"],
  ["MagicBrush human mask","78.4%","51.7%","+26.8%"],
  ["ground-truth region","59.4%","6.4%","+53.0%"]];
 tbl(s,r,{y:1.80,colW:[4.2,2.5,2.5,2.43],rowH:0.44,fontSize:11.5,centerCols:[1,2,3],hiRows:[2]});
 tint(s,M,4.15,CW,0.9);
 s.addText([{text:"We now beat the human annotation",options:{bold:true}},
  {text:"  —  +39.5% against +26.8%, reversing the one comparison the compositing test lost. And collateral 22.2% is "},
  {text:"below",options:{bold:true}},{text:" whole-frame editing's 24.2%, so we win on both quantities rather than trading one for the other. The oracle at +53.0% bounds what a perfect region would buy."}],
  {x:M+0.32,y:4.36,w:CW-0.64,h:0.62,fontFace:BODY,fontSize:11.5,color:INK,lineSpacing:16,margin:0});
 lab(s,M,5.20,10,"The operating point had to be found jointly");
 const g=[["","0 px","16 px","32 px","48 px","64 px"],
  ["thr 0.2","+31.2","+36.9","+36.7","+35.7","+32.7"],
  ["thr 0.3","+28.6","+36.5","+39.5","+37.9","+33.3"],
  ["thr 0.4","+26.5","+34.9","+38.4","+39.4","+35.9"],
  ["thr 0.5","+24.2","+32.8","+36.2","+38.4","+37.9"]];
 tbl(s,g,{y:5.48,colW:[1.63,2.0,2.0,2.0,2.0,2.0],rowH:0.30,fontSize:10,centerCols:[1,2,3,4,5],hiRows:[2]});
 s.addText("Optimal dilation rises with threshold — 16, 32, 48, 48 px. Sweeping dilation alone found a local optimum on one slice. The peak is interior on both axes.",
  {x:M,y:6.85,w:CW,h:0.4,fontFace:BODY,fontSize:10.5,color:MUTE,italics:true,margin:0});
 s.addNotes("This is the strongest downstream result and it came from re-tuning, not from a new model.\n\nThe first inpainting run reused 64 px of dilation from the compositing experiment, where an over-large mask costs nothing because it can only subtract. Under inpainting the mask decides what gets REGENERATED, so too much dilation destroys content that should have been preserved. Collateral came out at 39.1% — worse than whole-frame editing.\n\nSweeping threshold and dilation jointly, 20 cells, found 0.3 and 32 px: +39.5% net at 22.2% collateral. That is below whole-frame's 24.2%, so we beat the status quo on both quantities now.\n\nThe shape is the real finding. Every row has its own peak and the optimal dilation rises with the threshold. A higher threshold gives a smaller raw mask which needs more dilation to reach the same coverage, so the two trade along a ridge. That is exactly why sweeping dilation alone returned a local optimum.\n\nIf asked why not more samples: 60 per cell, one editor, one guidance setting. Say so.");}

/* 8 HONEST LIMITS */
{const s=slide("What we checked on ourselves","Every number here survived an attempt to break it",null);
 const r=[["Check","What it caught"],
  ["Contamination audit","an early headline trained on the evaluation split; retracted in the repository and re-run on disjoint shards"],
  ["Selection-bias audit","choosing the best epoch by dev IoU had inverted the v1/v2 conclusion; selection moved to a held-out slice of train"],
  ["Random-basis controls","two proposed improvements — free-space and geometric bases — failed against noise and were dropped"],
  ["Latency fairness","measured at 640 px against CLIPSeg’s 352 px, and only for v1 — re-measured matched and for the reported head: v2 is 10.3×"],
  ["Ceiling consistency","the headroom figure compared a train-set bound to a dev score; recomputed on dev"],
  ["Leakage audit","MagicBrush is multi-turn, so before scaling 3.65× we hashed every image on both sides — 0 of 794 dev images appear in train"],
  ["Threshold selection","dev peaks at 0.3, but adopting it would be test-set tuning; the threshold is chosen on train-val instead"]];
 tbl(s,r,{y:1.42,colW:[2.9,8.73],rowH:0.47,fontSize:9.5});
 tint(s,M,5.10,CW,0.95);
 s.addText("Two claims were retracted and two proposed improvements rejected by their own controls. That is why the three results that remain are worth stating.",
  {x:M+0.32,y:5.32,w:CW-0.64,h:0.6,fontFace:HEAD,fontSize:13,italics:true,color:INK,lineSpacing:17,margin:0});
 lab(s,M,6.25,10,"Known limitation, not yet quantified");
 s.addText("The remove ceiling on dev rests on 32 samples and carries uncertainty the figure does not show.",
  {x:M,y:6.50,w:CW,h:0.4,fontFace:BODY,fontSize:10.5,color:MUTE,italics:true,margin:0});
 s.addNotes("This slide is the credibility slide. Do not rush it.\n\n\"Every number in this deck survived an attempt to break it.\" Then pick two.\n\nThe contamination one: an overnight script merged dev into train. We caught it because a baseline scored 0.3743 against a training-time 0.1397 — a gap too big to be a metric difference. We retracted the headline and the beats-human-annotation claim, and re-ran on disjoint shards.\n\nThe selection one: choosing the best epoch by dev IoU had INVERTED the v1-versus-v2 conclusion. Not inflated it — inverted it.\n\nAlso worth a sentence: two of our own proposed improvements, the free-space and geometric bases, lost to a random control of equal size and were dropped.\n\nLand it: two claims retracted, two ideas rejected by their own controls. That is why the results that remain are worth stating.");}

/* 9 PLAN */
{const s=slide("What follows","Three months, and the bar each step must clear",null);
 const p=[["Month","Focus","Deliverable","Must beat"],
  ["1 · Sep–Oct","Supervision at scale — DONE","all 51 shards trained: 8,306 turns, 24 seeded runs, 0.1718 → 0.2065","0.1718 — cleared, p=4.1e-09"],
  ["2 · Oct–Nov","Features, not heads","exp08/11 show the head is not the insertion bottleneck — test text-conditioned features","a random-basis control of equal size"],
  ["3 · Nov–Dec","Editing and evaluation","mask-conditioned inpainting; collateral change outside the region — Review 2","MagicBrush mask at +18.5% net"]];
 tbl(s,p,{y:1.38,colW:[1.6,2.1,5.4,2.53],rowH:0.70,fontSize:10});
 lab(s,M,4.15,9,"The methodological bar this project already holds itself to");
 const b=[["Every claim reports its ablation","a config scoring IoU 0.1429 with delta −0.0012 was a blob predictor ignoring the instruction"],
  ["Every fix must beat its control","free-space and geometric bases both lost to random noise and were dropped"],
  ["Every split is disjoint by construction","train and dev come from different shard files, never merged"],
  ["Every figure regenerates from the data","python train/figures.py reads only the committed result JSONs"]];
 tbl(s,b,{y:4.43,colW:[3.9,7.73],rowH:0.40,fontSize:10});
 s.addNotes("Where this goes, and the bar each step has to clear.\n\nMonth 1 is DONE — that is this review. All 51 shards, 8,306 turns, 24 seeded runs, 0.1718 to 0.2065.\n\nMonth 2 is the real research problem and I want to be precise about it. We fitted coefficients directly to ground truth by least squares. That is an upper bound on what ANY predictor can do with these 32 prototypes: 0.212 for insertion, against 0.433 for removal. Insertion — the case this project exists for — has the lowest ceiling, because FastSAM's prototypes were trained to segment objects, and objects are not empty space.\n\nMore data cannot pass that. We are at 56 percent of it. So month 2 attacks the BASIS, not the head — and any new basis must beat a random basis of equal size, because that is the test our last attempt failed.\n\nMonth 3 is editing and evaluation for Review 2.");}

/* 10 THANK YOU */
{const s=pres.addSlide();s.background={color:PAPER};
 s.addText("Thank you",{x:M,y:0.78,w:CW,h:0.8,fontFace:HEAD,fontSize:38,bold:true,color:INK,margin:0});
 s.addText("We are happy to take questions.",{x:M,y:1.64,w:CW,h:0.4,fontFace:BODY,fontSize:15,color:MUTE,italics:true,margin:0});
 s.addShape(pres.ShapeType.rect,{x:M,y:2.88,w:CW,h:0.008,fill:{color:RULE}});
 s.addText([{text:"Amritha S",options:{bold:true}},{text:"   23BEC1368"}],{x:M,y:3.16,w:5.5,h:0.32,fontFace:BODY,fontSize:14,color:INK,margin:0});
 s.addText([{text:"Yugeshwaran P",options:{bold:true}},{text:"   23BEC1404"}],{x:M,y:3.53,w:5.5,h:0.32,fontFace:BODY,fontSize:14,color:INK,margin:0});
 s.addText("Guide  ·  Dr. Saranya M  (54783)\nSchool of Electronics Engineering  ·  VIT Chennai",{x:M,y:4.06,w:6.0,h:0.7,fontFace:BODY,fontSize:12,color:MUTE,lineSpacing:17,margin:0});
 lab(s,W-M-4.55,3.18,4.3,"Everything reproduces",{color:MUTE});
 s.addText("github.com/Amritha902/edit-region-prediction\n\n12 experiments  ·  seeded  ·  runs on a MacBook Air",{x:W-M-4.55,y:3.46,w:4.3,h:1.0,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 s.addNotes("Closing.\n\n\"Everything reproduces. One repository, 13 experiments, every result seeded, every figure regenerated from committed JSON. It runs on a MacBook.\"\n\nThen stop talking and take questions.\n\nLIKELY QUESTIONS, short answers:\n\nWhy not compare to AdaptEdit directly? It reports no mask IoU and we cannot run 20 billion parameters. That absence is the gap we fill.\n\nIs 0.2065 good? Against the right references, yes — it beats a 150M pretrained segmenter and the dataset's own human annotation. Against the ceiling, we are at about 56 percent, and we say so.\n\nWhy is insertion still weak? A measured representational bound, not a training failure. We have the least-squares proof.\n\nWhy a frozen backbone? It buys 32 times fewer trainable parameters and 10 times lower latency. It also caps accuracy — that is a stated trade, not an oversight.");}

pres.writeFile({fileName:"ISE_Review1_Final.pptx"}).then(f=>console.log("wrote",f));
