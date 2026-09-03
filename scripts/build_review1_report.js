const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, LevelFormat, convertInchesToTwip
} = require("docx");

const CW = 9746;                       // content width in DXA (A4, 0.75" margins)
const GREY = "F2F2F2";
const SERIF = "Cambria";

// ---------- helpers ----------
const P = (text, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after === undefined ? 120 : o.after, line: o.line || 276 },
  indent: o.indent,
  children: [new TextRun({ text, font: SERIF, size: o.size || 21, bold: o.bold, italics: o.italics, color: o.color })]
});

const RICH = (runs, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after === undefined ? 120 : o.after, line: o.line || 276 },
  children: runs.map(r => new TextRun({
    text: r.t, font: SERIF, size: r.size || o.size || 21,
    bold: r.b, italics: r.i, color: r.c
  }))
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 320, after: 160 },
  children: [new TextRun({ text, font: SERIF, size: 26, bold: true, color: "1A1A1A" })]
});

const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 220, after: 110 },
  children: [new TextRun({ text, font: SERIF, size: 22, bold: true, color: "1A1A1A" })]
});

const BULLET = (text) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { after: 80, line: 276 },
  children: [new TextRun({ text, font: SERIF, size: 21 })]
});

const CAPTION = (text) => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { before: 80, after: 240 },
  children: [new TextRun({ text, font: SERIF, size: 18, italics: true, color: "555555" })]
});

function cell(text, w, o = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: o.head ? { type: ShadingType.CLEAR, fill: GREY, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: String(text).split("\n").map(line => new Paragraph({
      alignment: o.align || AlignmentType.LEFT,
      spacing: { after: 0, line: 240 },
      children: [new TextRun({ text: line, font: SERIF, size: o.size || 17, bold: o.head || o.bold })]
    }))
  });
}

function table(widths, rows, opts = {}) {
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    borders: {
      top:    { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" },
      left:   { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" },
      right:  { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "DDDDDD" },
      insideVertical:   { style: BorderStyle.SINGLE, size: 2, color: "DDDDDD" }
    },
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => cell(c, widths[j], {
        head: i === 0,
        size: opts.size,
        align: (opts.centerCols || []).includes(j) ? AlignmentType.CENTER : AlignmentType.LEFT
      }))
    }))
  });
}

// ================= TITLE PAGE =================
const title = [
  new Paragraph({ spacing: { before: 900, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "VELLORE INSTITUTE OF TECHNOLOGY, CHENNAI", font: SERIF, size: 28, bold: true })] }),
  new Paragraph({ spacing: { after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "School of Electronics Engineering", font: SERIF, size: 22 })] }),
  new Paragraph({ spacing: { after: 700 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "FOUNDATIONS OF DATA SCIENCE", font: SERIF, size: 20, color: "555555" })] }),

  new Paragraph({ spacing: { after: 60 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "PROJECT REPORT — REVIEW 1", font: SERIF, size: 20, bold: true, color: "555555" })] }),
  new Paragraph({ spacing: { after: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "INSTRUCT-SEG-EDIT", font: SERIF, size: 44, bold: true })] }),
  new Paragraph({ spacing: { after: 800 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Disentangling Localization from Editing in Instruction-Guided Image Manipulation", font: SERIF, size: 24, italics: true })] }),

  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Submitted by", font: SERIF, size: 20, color: "555555" })] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "AMRITHA S", font: SERIF, size: 24, bold: true })] }),
  new Paragraph({ spacing: { after: 140 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "23BEC1368  |  amritha.s2023@vitstudent.ac.in", font: SERIF, size: 20 })] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "YUGESHWARAN P", font: SERIF, size: 24, bold: true })] }),
  new Paragraph({ spacing: { after: 700 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "23BEC1404", font: SERIF, size: 20 })] }),

  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Submitted to", font: SERIF, size: 20, color: "555555" })] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Dr. SARANYA M", font: SERIF, size: 24, bold: true })] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Employee ID: 54783  |  School of Computer Science and Engineering (SCOPE)", font: SERIF, size: 20 })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Faculty, Foundations of Data Science", font: SERIF, size: 20 })] }),
  new Paragraph({ spacing: { after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Academic Year 2025 – 2026", font: SERIF, size: 20, color: "555555" })] }),
  new Paragraph({ children: [new PageBreak()] })
];

// ================= ABSTRACT =================
const abstract = [
  H1("ABSTRACT"),
  P("This report presents Instruct-Seg-Edit, a two-stage image editing system that separates the decision of where an edit belongs from the execution of the edit itself. The localization stage is an instance segmentation network derived from the YOLOv8 family: a CSP-inspired backbone with C2f modules, a PAN-style neck, and a YOLACT-style prototype and coefficient segmentation head, conditioned on the edit instruction through a CLIP text encoder. At 71.8 million parameters the model stays efficient and reaches inference speeds close to the YOLO detectors it derives from."),
  P("The motivation for building this stage is that diffusion editing models change parts of an image the user never asked about. Inpainting models fix that leakage by restricting generation to a mask, but they cannot work the mask out on their own and must be given one. Our localization model fills that gap. It reads the image together with the edit instruction and returns the mask the inpainting stage consumes, so the two halves of the problem are handled by two separate models rather than one entangled network."),
  P("No public dataset provides the mask supervision this requires. Segmentation corpora such as MSCOCO-seg annotate the objects a phrase refers to, which is not the same region an edit changes: for the instruction “put a hat on the man” the referent is the man, but the region that must change is the empty space above his head. The training set is therefore derived. Starting from the PixelProse RedCaps corpus, edit instructions are generated with a vision-language model, edited images are produced with an instruction-following diffusion editor, and ground-truth edit masks are recovered from the perceptual difference between the original and edited images in CIE L*a*b* space followed by morphological cleanup. MSCOCO-seg is retained in a strictly auxiliary role, to pre-train the backbone and neck on detection before instruction conditioning is introduced."),
  P("This report covers the model architecture and its full parameter breakdown, a literature survey of sixteen works with a quantitative relevance comparison, the five research gaps that follow from it, the dataset generation protocol, and an exploratory analysis of 1,290,861 source records."),
  RICH([{ t: "Index Terms — ", b: true }, { t: "Instance segmentation, YOLOv8, prototype masks, distribution focal loss, efficient segmentation, inpainting, instruction-guided image editing, CLIP, mask-conditioned diffusion, derived datasets." }], { after: 200 })
];

// ================= 1. INTRODUCTION =================
const intro = [
  H1("1.  INTRODUCTION"),
  P("Diffusion-based image editing models are popular because they give access to realistic edits without requiring skill of the user. They are also notorious for making changes outside what the instruction asked for. Inpainting models address this by confining generation to a masked region, typically denoising the masked area under a ControlNet or a U-Net. Although inpainting models follow instructions well and generate high quality edits inside the region they are given, they cannot identify that region themselves and so require a mask as input. This leaves an opening for a segmentation model trained to read the instruction, identify the part of the image that has to change, and emit the mask the inpainting model needs."),
  P("Instance segmentation provides a pixel-level mask, and optionally a class label, for each object in an image. It combines the localization strength of object detection with pixel-level masking, which is exactly the form of output an inpainting model consumes. Most modern models are either large and accurate at the cost of inference speed, or small and fast at the cost of quality. That leaves room for a segmentation model balancing the two. We present an efficient model of 71.8 million parameters that delivers good segmentation quality for its size, deeply inspired by the YOLOv8 architecture, particularly its backbone and neck."),

  H2("1.1  Problem Statement"),
  P("Current instruction-based editing models such as InstructPix2Pix learn localization and editing implicitly. Both decisions are absorbed into a single network, which leads to unintended modifications outside the desired region and gives the user very little control. The core challenge is this: how do we explicitly separate the decision of where to edit from the execution of what to edit, and let the user step in and correct the first before the second ever runs?"),

  H2("1.2  Objectives"),
  BULLET("Derive a dataset of image, edit instruction, edited image and edit mask, since no public dataset provides the edit mask needed for supervision."),
  BULLET("Build a localization model that predicts the edit mask from an image and an instruction, at interactive speed."),
  BULLET("Condition a diffusion editing model on that mask, with the reconstruction loss computed only inside the masked region."),
  BULLET("Expose the predicted mask so the user can refine it before the editing stage runs."),
  BULLET("Evaluate localization with IoU and Dice, and editing with LPIPS, FID and CLIP similarity, against an end-to-end InstructPix2Pix baseline."),

  H2("1.3  Contributions"),
  BULLET("Disentangled control. Localization and editing are separated into two models rather than learned jointly inside one denoiser."),
  BULLET("Supervised localization. The edit region receives direct ground truth supervision from recovered edit masks, instead of being left to emerge on its own inside an attention map."),
  BULLET("Human-in-the-loop interactivity. The predicted edit region is shown to the user and can be corrected before generation."),
  BULLET("Interpretable editing. The mask is an explicit picture of where the model believes the edit belongs."),
  BULLET("A practical instance segmentation system inspired by YOLOv8, pre-trained on detection and fine-tuned for instruction-conditioned segmentation, with a full parameter breakdown for reproducibility.")
];

// ================= 2. RELATED WORK =================
const litRows = [
  ["#", "Work (Author, Year)", "Technique", "Reported Performance", "Research Gap Identified", "Rel."],
  ["1", "Mask R-CNN\nHe et al., 2017", "Two-stage detector with a parallel mask branch and RoIAlign", "35.7 mask AP (COCO test-dev); ~5 FPS", "Accepts no language input at all, so the mask cannot follow an instruction. Two-stage cost holds it near 5 FPS, too slow for an interactive editing loop.", "0.589"],
  ["2", "YOLACT\nBolya et al., 2019", "Prototype masks plus per-instance coefficients, single-shot", "29.8 mask AP @ 33.5 FPS", "Real-time but purely visual. Masks are tied to object categories rather than to any edit intent, and mask precision is lower than two-stage methods.", "0.612"],
  ["3", "SOLOv2\nWang et al., 2020", "Proposal-free grid-wise instance prediction with dynamic convolution", "38.8 mask AP; 31.0 AP @ 31.3 FPS", "Strong proposal-free segmentation, but there is no mechanism anywhere in the architecture to condition the output on text.", "0.612"],
  ["4", "Mask2Former\nCheng et al., 2022", "Masked-attention transformer for universal segmentation", "50.1 mask AP (Swin-L)", "Highest accuracy in the survey, but high VRAM and slow inference make it unusable in a human-in-the-loop editor, and it is not language-conditioned.", "0.500"],
  ["5", "YOLOv8-seg\nJocher et al., 2023", "C2f backbone, PAN neck, anchor-free decoupled head with DFL, prototype mask head", "30.5 to 43.4 mask AP (n to x scales)", "The best speed-accuracy balance available and our chosen backbone, but it segments a fixed set of object classes and has no notion of an edit instruction.", "0.612"],
  ["6", "CLIPSeg\nLüddecke, Ecker, 2022", "Frozen CLIP with a lightweight transformer decoder, text prompts", "~43.4 mIoU on PhraseCut", "Segments what a phrase refers to, not what an edit would change. Output is semantic rather than instance-level, so overlapping objects are not separated.", "0.686"],
  ["7", "LAVT\nYang et al., 2022", "Early language-vision fusion inside the visual encoder", "72.73 oIoU on RefCOCO val", "Trained on referring expressions, which name an object rather than describe a change. It performs no editing, so the mask has no downstream consumer.", "0.671"],
  ["8", "SAM\nKirillov et al., 2023", "Promptable segmentation, 1B-mask pre-training, zero-shot transfer", "Strong zero-shot transfer; geometric prompts", "Native prompts are points and boxes, not language, so text conditioning needs a separate grounding model bolted on. It produces no edit.", "0.822"],
  ["9", "Grounded-SAM\nRen et al., 2024", "GroundingDINO text grounding chained with SAM mask decoding", "Open-vocabulary zero-shot grounding", "Reaches text-to-mask only by chaining two large frozen models, which is heavy and not trainable end to end. It still grounds on object nouns, not on edit intent.", "0.833"],
  ["10", "LISA\nLai et al., 2024", "Multimodal LLM coupled with a segmentation decoder", "gIoU on ReasonSeg", "Handles reasoning well but is 7B+ parameters and slow. It still targets the referred object rather than the edit region, and performs no generation.", "0.707"],
  ["11", "InstructPix2Pix\nBrooks et al., 2023", "Instruction-conditioned diffusion trained on synthetic pairs", "CLIP similarity / CLIP direction", "Our baseline. Localization is entirely implicit inside the denoising U-Net, so edits leak outside the intended region and the user can neither see nor correct the spatial decision.", "0.612"],
  ["12", "Latent Diffusion Inpainting\nRombach et al., 2022", "Latent-space diffusion with mask conditioning", "FID / LPIPS", "Confines the edit correctly, but the mask has to be supplied by the user. The model cannot derive the region from the instruction.", "0.686"],
  ["13", "ControlNet\nZhang et al., 2023", "Auxiliary conditioning branch on a frozen diffusion backbone", "FID / CLIP score", "Spatial conditioning is flexible, but the control signal is an input it is handed, not something inferred from the instruction.", "0.686"],
  ["14", "DiffEdit\nCouairon et al., 2023", "Mask inferred by contrasting noise estimates under source and target text", "LPIPS / FID", "Closest existing work. The mask is an inference-time by-product of diffusion: never supervised, not separately trainable or evaluable, not instance-aware, and never shown to the user.", "0.791"],
  ["15", "Prompt-to-Prompt\nHertz et al., 2023", "Cross-attention map manipulation for controlled editing", "Qualitative and CLIP score", "Cross-attention maps act as an implicit localization signal, but they are coarse and unsupervised and cannot be thresholded into a precise mask.", "0.756"],
  ["16", "PixelProse\nSingla et al., 2024", "16M synthetically generated dense captions over web images", "Corpus resource", "OUR SOURCE CORPUS. Supplies images with both a human caption and a dense VLM caption, but carries no edit instruction, no edited image and no edit mask. All three are derived on top of it.", "—"],
  ["17", "MS COCO / MSCOCO-seg\nLin et al., 2014", "Detection, instance segmentation and captioning benchmark", "118,287 train / 5,000 val; defines mask AP", "AUXILIARY ONLY. Its instance masks annotate the object a phrase refers to, not the region an edit changes, so it cannot supervise the edit-region objective. Used to pre-train the backbone and neck on detection.", "0.500"]
];

const gapRows = [
  ["#", "Research Gap", "Proposed Technique / Contribution"],
  ["G1", "Instruction-guided editors learn localization implicitly inside the denoising U-Net, so edits leak outside the intended region and the spatial decision cannot be inspected.", "A two-stage architecture in which a dedicated localization model predicts the edit mask before any generation happens, making that decision explicit and separable."],
  ["G2", "Inpainting and mask-conditioned models produce controlled edits but need the mask as an input and cannot derive it from an instruction.", "An instruction-conditioned localization network that generates exactly the mask the inpainting stage consumes, removing manual annotation from the workflow."],
  ["G3", "Text-conditioned segmentation models are trained on referring expressions, which describe what a phrase denotes rather than what an instruction would change.", "Direct ground truth supervision using edit masks recovered from before-and-after image pairs, so the model is trained on the edit-region objective itself."],
  ["G4", "No public dataset provides the image, instruction, edited image and edit mask quadruple needed to supervise this objective. Segmentation corpora annotate referents, not edit regions.", "An automated generation pipeline using a vision-language model for instructions, an instruction-following diffusion editor for edited images, and perceptual differencing in L*a*b* with morphological cleanup for masks."],
  ["G5", "Existing pipelines give the user no point of intervention between interpretation and generation. The only control available is rewording the prompt.", "A human-in-the-loop interface that shows the predicted mask and allows the user to correct it before the diffusion stage runs."]
];

const related = [
  H1("2.  RELATED WORK"),
  P("The work relevant to this project falls into four groups: instance segmentation, which gives us the mask machinery; text-conditioned segmentation, which links language to regions; instruction editing, which defines the task; and mask-conditioned generation, which forms the second stage."),

  H2("2.1  Two-Stage Segmentation"),
  P("Mask R-CNN introduced box-level proposals and mask heads for instance segmentation, producing high accuracy at the cost of two-stage complexity. It reports 35.7 mask AP on COCO test-dev with a ResNet-101-FPN backbone and remains a standard accuracy reference. Its drawback for our purpose is speed. Computing a mask separately for every proposal holds it near 5 FPS, which is too slow to sit inside an interactive editing loop."),

  H2("2.2  Single-Shot Segmentation"),
  P("Anchor-free detectors and prototype-based systems such as YOLACT and SOLO demonstrated real-time instance segmentation by decoupling heavy per-instance computation. YOLACT predicts a set of image-wide prototype masks along with a short coefficient vector per instance, and recovers each mask as a linear combination of the prototypes. The expensive convolution work is shared across all instances, so the per-instance cost is very small. It reports 29.8 mask AP at 33.5 FPS. SOLOv2 later reached 38.8 mask AP by predicting instances directly over spatial grids without proposals. We use the prototype and coefficient formulation in our own head, because editing scenes often contain many candidate objects while only one is finally edited."),

  H2("2.3  YOLO Family"),
  P("YOLOv8 refines CSP-inspired backbones, PAN neck feature fusion, and decoupled anchor-free heads with Distribution Focal Loss to improve localization precision while maintaining high throughput. DFL predicts a discrete distribution over each box coordinate rather than a single value, which sharpens localization. The segmentation variants attach a YOLACT-style prototype head to this backbone and span 30.5 to 43.4 mask AP on COCO across model scales. Our method sits at the intersection of single-shot detectors and prototype mask segmentation, adopting YOLOv8 design choices and the prototype plus coefficients formulation for efficiency."),

  H2("2.4  Text-Conditioned and Promptable Segmentation"),
  P("A separate line of work links language to segmentation. CLIPSeg adds a light transformer decoder on top of a frozen CLIP backbone to segment from a text prompt. LAVT fuses language into the visual encoder early rather than late and performs strongly on referring expression segmentation, reaching 72.73 oIoU on RefCOCO. The Segment Anything Model brought large-scale promptable segmentation with good zero-shot behaviour, but its native prompts are points and boxes, so text conditioning needs an external grounding model, as in Grounded-SAM. LISA pairs a multimodal LLM with a segmentation decoder for instructions that need reasoning."),
  RICH([
    { t: "There is an important limitation shared by all of these, and it is the observation the whole project rests on. " },
    { t: "They segment what an expression refers to, not what an edit would change.", b: true },
    { t: " If the instruction is “put a hat on the man”, the phrase refers to the man, but the region that actually has to change is the empty space above his head. A referring-expression model will not return it. The same argument disqualifies segmentation corpora as a source of supervision, because their masks annotate referents." }
  ]),

  H2("2.5  Instruction-Guided Editing and Mask-Conditioned Generation"),
  P("InstructPix2Pix showed that an instruction-following editor can be trained on synthetic pairs, and we use it as the generator when building our dataset. Latent diffusion inpainting and ControlNet give high quality mask-conditioned generation but treat the conditioning signal as an input they are handed. DiffEdit is the closest existing work to ours, since it also produces an edit mask automatically, by contrasting the noise predictions obtained under the source and the target text. But that mask is a by-product of the diffusion process itself. It is computed at inference time, it is never supervised, it cannot be trained or evaluated separately, and it is not shown to the user for correction. Prompt-to-Prompt works in a similar spirit by manipulating cross-attention maps as an implicit localization signal, though those maps are coarse."),

  H2("2.6  Literature Survey Summary"),
  P("Table 1 lists every surveyed work with its technique, its reported performance, and the specific research gap it leaves open for our problem. The relevance score in the final column measures how much of our required capability set each work already covers."),
  P("Eight capability dimensions were defined, each a property this project requires: instruction conditioning, pixel-level mask output, instance-level reasoning, real-time efficiency, image editing, diffusion-based generation, explicit separation of localization from editing, and human-in-the-loop refinement. Every surveyed work is encoded as a vector over these eight dimensions, scoring 1 for a fully supported capability, 0.5 for partial support and 0 for none. Our own project targets all eight, so its vector is all ones. The score is the cosine of the angle between the two vectors:"),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 140 },
    children: [new TextRun({ text: "cos(θ)  =  (V_work · V_ours) / ( ‖V_work‖ × ‖V_ours‖ )", font: SERIF, size: 21, italics: true })] }),
  P("As a worked example, DiffEdit is encoded as [1, 1, 0, 0, 1, 1, 1, 0]. Its dot product with our all-ones vector is 5, its norm is the square root of 5 which is 2.236, and our norm is the square root of 8 which is 2.828. The cosine is therefore 5 divided by 6.325, giving 0.791. A score near 1 means the work already covers most of what we need and is a close competitor, while a low score means it contributes one component but not the overall capability."),
  table([420, 1450, 1750, 1500, 3826, 800], litRows, { size: 15, centerCols: [0, 5] }),
  CAPTION("Table 1: Literature survey with the research gap identified in each work. Relevance is the cosine similarity of each work to this project over the eight-dimension capability vector. Performance figures are as reported in the original papers."),

  H2("2.7  Research Gaps and Proposed Techniques"),
  P("The highest relevance scores are Grounded-SAM at 0.833, SAM at 0.822 and DiffEdit at 0.791. None reaches the full capability set, and the ones that come closest do so on different grounds. Grounded-SAM and SAM reach text-to-mask only by chaining large frozen models together and perform no edit at all, while DiffEdit performs the edit but never supervises or exposes its mask. Five gaps follow."),
  table([620, 4400, 4726], gapRows, { size: 16, centerCols: [0] }),
  CAPTION("Table 2: Research gaps identified from the literature and the corresponding proposed techniques.")
];

// ================= 3. METHODOLOGY =================
const paramRows = [
  ["Category", "Module", "Component", "Parameters"],
  ["Backbone", "model.0", "Conv", "2,321"],
  ["", "model.1", "Conv", "115,681"],
  ["", "model.2", "C2f", "708,323"],
  ["", "model.3", "Conv", "461,761"],
  ["", "model.4", "C2f", "4,217,126"],
  ["", "model.5", "Conv", "1,844,481"],
  ["", "model.6", "C2f", "11,850,726"],
  ["", "model.7", "Conv", "3,687,681"],
  ["", "model.8", "SPPF", "7,618,083"],
  ["", "", "Backbone subtotal", "30,506,183"],
  ["Neck", "model.9", "C2f", "2,460,323"],
  ["", "model.12", "C2f", "8,603,046"],
  ["", "model.15", "C2f", "2,130,563"],
  ["", "model.16", "Conv", "922,561"],
  ["", "model.18", "C2f", "7,374,246"],
  ["", "model.19", "Conv", "3,687,681"],
  ["", "model.21", "C2f", "8,603,046"],
  ["", "", "Neck subtotal", "33,781,466"],
  ["Segmentation Head", "model.22.proto", "Mask prototype branch", "2,267,009"],
  ["", "model.22.dfl", "Distribution focal loss conv", "16"],
  ["", "model.22.cv2", "Box prediction branch", "1,257,024"],
  ["", "model.22.cv3", "Class prediction branch", "4,611,523"],
  ["", "model.22.cv4", "Mask coefficient branch", "1,200,736"],
  ["", "", "Segmentation head subtotal", "9,336,308"],
  ["", "", "TOTAL PARAMETERS", "71,812,301"]
];

const method = [
  H1("3.  METHODOLOGY"),
  P("The pipeline follows a two-stage strategy. Stage 1 takes the original image and the edit instruction and predicts a binary edit mask. The mask is optionally corrected by the user, then used to gate the image. Stage 2 takes the image, the instruction and the mask, and generates the edited image under a loss evaluated only inside the masked region."),

  H2("3.1  Stage 1 — Localization Model"),
  RICH([{ t: "Input: ", b: true }, { t: "original image and edit instruction.    " }, { t: "Output: ", b: true }, { t: "edit mask locating the region to be edited." }]),
  BULLET("Image encoder: a pre-trained Vision Transformer (ViT-B/16) or CLIP image encoder."),
  BULLET("Text encoder: the CLIP text encoder for the instruction and caption embeddings. Other embedding models could be used, but CLIP is trained for image conditioning, which makes it close to ideal here."),
  BULLET("Fusion module: a hybrid attention mechanism using both self-attention and cross-attention to fuse the image and text latents."),
  BULLET("Backbone and neck: CSP-inspired convolution stages with C2f modules for efficient gradient flow, and a PAN-style FPN with a bottom-up path that fuses multi-scale features."),
  BULLET("Segmentation head: a prototype branch, a small FCN with upsampling that outputs K prototype masks at near-input resolution, and a coefficient branch predicting the coefficients used to linearly combine them. K is 32 by default."),
  BULLET("Segmentation decoder: a U-Net style decoder with skip connections producing the predicted edit mask, thresholded at 0.5."),

  H2("3.2  Interactive Refinement"),
  P("Between the two stages the predicted mask is rendered as an overlay and offered to the user for correction. This is the point of intervention that end-to-end editors do not have. If the model has picked the wrong object, the fix is a direct local edit to the mask rather than a guess at a better prompt, and the corrected mask is what the diffusion stage receives."),

  H2("3.3  Stage 2 — Mask-Conditioned Diffusion Model"),
  RICH([{ t: "Input: ", b: true }, { t: "original image, edit instruction and the predicted or refined mask.    " }, { t: "Output: ", b: true }, { t: "edited image." }]),
  BULLET("Base model: an adapted Pix2Pix HD or latent diffusion model."),
  BULLET("Mask conditioning: the mask is concatenated as an additional input channel, giving a 4-channel input of R, G, B and Mask."),
  BULLET("Masked loss: the reconstruction loss is computed only within the masked region, so edit quality is enforced exactly where the edit is wanted and the objective cannot reward changes elsewhere."),

  H2("3.4  Loss Functions"),
  BULLET("Box regression: CIoU with DFL refinement."),
  BULLET("Objectness and classification: BCE with logits."),
  BULLET("Mask loss: BCE, Dice or a focal variant, computed between the predicted per-instance mask and the ground truth mask after cropping to the bounding box."),
  BULLET("Multi-task weighting: an empirical schedule with a detection to segmentation ratio of 1:2, used to stop the segmentation head from overwhelming the detection pre-trained weights during fine-tuning."),

  H2("3.5  Parameter Distribution"),
  P("Table 3 gives the parameter count of every module in the localization model, included for interpretability and approximate resource planning. The prototype branch produces 32 high-resolution shared masks by default and the coefficient branch predicts a 32-value vector per instance, so per-instance cost stays small. The total of 71,812,301 parameters is consistent with the YOLOv8x-seg scale the architecture derives from, and is corroborated by the trained checkpoint itself: its tensor storage is 287,128,846 bytes, which at fp32 is 71,782,212 parameters, agreeing to within 0.04 percent. Note that the per-module rows sum to 73,623,957; that discrepancy of 1,811,656 against the measured total has not yet been traced, and the measured figure should be treated as authoritative."),
  table([1900, 1700, 3646, 2500], paramRows, { size: 16, centerCols: [3] }),
  CAPTION("Table 3: Parameter distribution of the localization model. The measured total is 71.8 million parameters, confirmed against the trained checkpoint."),

  H2("3.6  Evaluation Protocol"),
  BULLET("Localization: Intersection over Union (IoU), Dice coefficient, and precision and recall of the predicted edit region."),
  BULLET("Editing: LPIPS inside the masked region, FID for overall image quality, and CLIP similarity between the edited image and the instruction."),
  BULLET("Efficiency: inference latency and frames per second at batch size 1."),
  BULLET("Baseline: end-to-end InstructPix2Pix, to isolate the benefit of the explicit separation.")
];

// ================= 4. DATASET =================
const schemaRows = [
  ["Field", "Type", "Origin", "Role in Training"],
  ["Image", "RGB, 512 × 512", "PixelProse / RedCaps (public)", "Input to Stage 1 and Stage 2"],
  ["Original caption", "Text", "PixelProse / RedCaps (public)", "Context for instruction generation"],
  ["VLM caption", "Text", "Gemini 1.0 Pro Vision (in corpus)", "Context for instruction generation"],
  ["Edit instruction", "Text", "Vision-language model (derived)", "Conditioning input to both stages"],
  ["Edited image", "RGB", "Instruction-following diffusion editor (derived)", "Target for Stage 2"],
  ["Edit mask", "Binary mask", "L*a*b* ΔE + morphology (derived)", "Target for Stage 1"]
];

const edaRows = [
  ["Property", "Original caption", "VLM caption"],
  ["Mean length (characters)", "51.9", "493.6"],
  ["Median length (characters)", "39.0", "442.0"],
  ["Mean length (words)", "9.5", "92.1"],
  ["Median length (words)", "7.0", "83.0"],
  ["Mean unique words", "9.0", "53.9"],
  ["Vocabulary size", "166,237", "151,215"],
  ["Missing values", "0", "0"]
];

const cocoRows = [
  ["Property", "Value"],
  ["Dataset", "MS COCO 2017, instance segmentation split (MSCOCO-seg)"],
  ["Training images", "118,287"],
  ["Validation images", "5,000"],
  ["Object categories", "80 thing categories"],
  ["Annotated instances", "Approximately 860,000 in train2017"],
  ["Annotation format", "JSON; polygon vertex lists, RLE for crowd regions"],
  ["Role in this project", "Detection pre-training of the backbone and neck only"]
];

const dataset = [
  H1("4.  DATASET DESCRIPTION"),

  H2("4.1  Why the Dataset Has to Be Derived"),
  P("Supervising Stage 1 requires four fields per record: the original image, the edit instruction, the edited image, and the edit mask. Public corpora supply the image and its captions and stop there. The remaining three are manufactured."),
  P("It is worth being precise about why an existing segmentation corpus cannot be substituted, because it is the most natural objection to this design. MSCOCO-seg provides high-quality instance masks over 80 categories, and each image carries five human captions, so it superficially appears to offer both the language and the spatial target. But a COCO instance mask annotates the object a noun phrase refers to. As established in Section 2.4, that is not the region an edit changes. Training on COCO instance masks would teach the model to reproduce exactly the referring-expression behaviour this project identifies as the gap. The edit mask must therefore be recovered empirically, from the observed difference between an image before and after a real edit."),

  H2("4.2  Generation Pipeline"),
  BULLET("Source images. The PixelProse corpus from Hugging Face, a collection of over 16 million synthetically generated captions produced with Gemini 1.0 Pro Vision. We use the RedCaps subset. Each record supplies an image URL, an original human caption and a dense VLM caption."),
  BULLET("Instruction generation. A vision-language model reads the image together with both captions and emits a single concrete edit instruction, constrained to a tangible change to the primary subject. It is run locally through an OpenAI-compatible endpoint."),
  BULLET("Image editing. An instruction-following diffusion editor applies the generated instruction to the source image. InstructPix2Pix and FLUX.1 Kontext have both been implemented; both are small enough for local batched inference."),
  BULLET("Ground-truth masks. The edit mask is recovered from the perceptual difference between the original and edited images, as described in Section 4.3."),

  H2("4.3  Edit Mask Extraction"),
  P("The mask is computed as the CIE76 colour difference between the original and edited images in CIE L*a*b* space, thresholded at ΔE > 12, followed by a morphological opening and closing with an elliptical structuring element and removal of connected components below 40 pixels."),
  P("Differencing in L*a*b* rather than in grayscale is a deliberate choice. A grayscale difference discards hue, so an edit that changes colour at near-constant luminance produces almost no signal: a mid-tone red-to-blue change measures a grayscale delta of 26, below a threshold of 30, and would be discarded as though the editor had ignored the instruction. The same edit measures ΔE 105 in L*a*b*. Since the corpus is dominated by colour descriptors, as Section 4.6 shows, this failure mode would have removed a substantial and systematically biased portion of the data."),
  P("The threshold was selected against simulated whole-image regeneration noise, because diffusion editors redraw the entire frame and the threshold's practical job is rejecting that background. Without morphological cleanup, a threshold of ΔE > 8 reports 13.0 percent of the frame as edited at a noise level of σ = 5 against a ground truth of 3.39 percent. With cleanup, every threshold tested holds at approximately 3.47 percent even at σ = 8, which is what permits the sensitive threshold of 12."),

  H2("4.4  Dataset Schema"),
  table([1900, 1500, 3346, 3000], schemaRows, { size: 16 }),
  CAPTION("Table 4: Dataset schema. The first three fields come from the public corpus; the last three are derived."),

  H2("4.5  Auxiliary Dataset for Pre-training"),
  P("MSCOCO-seg is used to pre-train the backbone and neck before instruction conditioning is introduced. Detection pre-training gives the backbone a strong initialisation, after which the prototype mask head and the fusion module are fine-tuned on the derived dataset. Weights for the backbone and neck are loaded from the detection checkpoint and the segmentation head is initialised randomly. Its role is strictly auxiliary; no edit-region supervision comes from it."),
  table([3400, 6346], cocoRows, { size: 16 }),
  CAPTION("Table 5: MSCOCO-seg, used only for detection pre-training of the backbone and neck."),

  H2("4.6  Exploratory Data Analysis"),
  P("The source corpus was analysed over 1,290,861 RedCaps records spanning 28 columns, with no missing values in either caption field. Table 6 summarises the two caption types."),
  table([4000, 2873, 2873], edaRows, { size: 16, centerCols: [1, 2] }),
  CAPTION("Table 6: Caption statistics over 1,290,861 records of the PixelProse RedCaps subset."),
  P("The most consequential result is the relationship between the two caption types. The TF-IDF cosine similarity between an image's original caption and its VLM caption, measured over a 1,000-record sample, has a mean of 0.174 and a median of 0.053, with a standard deviation of 0.221 and a range of 0.000 to 0.834. A median of 0.053 means that more than half of all image pairs share essentially no vocabulary between their two captions. Vocabulary overlap across the whole corpus is 0.404, with 74,927 tokens appearing only in original captions and 59,905 only in VLM captions."),
  P("This directly justifies a design decision in the generation pipeline. The instruction generator receives the concatenation of both captions rather than either one alone, and the analysis shows that this is not redundancy: the two fields carry near-disjoint information. The original caption supplies the human's framing, while the VLM caption supplies the dense visual description, including proper nouns read incidentally off signage and packaging."),
  P("Word-frequency analysis shows the VLM captions are dominated by colour and spatial descriptors, with white appearing in 760,511 captions, red in 623,301, black in 581,144 and green in 470,619. This distribution is the empirical reason the mask extraction in Section 4.3 must be sensitive to hue."),

  H2("4.7  Preprocessing and Filtering"),
  BULLET("Images are resized with aspect-preserving padding. 640 × 640 is used for detection pre-training and 800 × 800 for segmentation fine-tuning, the higher resolution chosen to improve mask detail."),
  BULLET("Augmentation during pre-training: mosaic, MixUp, random horizontal flip, HSV colour jitter, random scale and translate. During fine-tuning: horizontal flip, crop, colour jitter and synthetic box jitter."),
  BULLET("Predicted masks are thresholded at 0.5. Connected components below 40 pixels are discarded as generation noise."),
  BULLET("Records are filtered out when the recovered mask is degenerate. A near-empty mask means the editor ignored the instruction, and a near-full mask means it applied a global style change instead of a local edit. Neither provides a usable localization target.")
];

// ================= 5. STATUS =================
const status = [
  H1("5.  IMPLEMENTATION STATUS"),
  P("The dataset generation pipeline is implemented end to end. Instruction generation runs asynchronously and in batches against a local vision-language endpoint and is resumable, keyed by a hash of the source URL. Edit generation is driven over a WebSocket against a local ComfyUI server running a FLUX.1 Kontext workflow, with an InstructPix2Pix path implemented as an alternative. Mask extraction is implemented as described in Section 4.3 and is shared by the notebooks and the automated tests, so the tested code path is the one the pipeline executes."),
  P("Verification is automated and requires no dataset. On a synthetic pair whose true edit region occupies 3.39 percent of the frame, the extractor recovers 3.47 percent; on a colour-only edit whose true region is 5.19 percent it recovers 5.28 percent, where the grayscale method it replaced recovers 0.00 percent. A prototype promptable segmenter builds, completes a forward pass and completes a training step."),
  P("Work remaining for Review 2, in order: complete generation over the sampled records; add the CLIP text-conditioning pathway to the localization model so that it is instruction-conditioned rather than point-prompted; pre-train on MSCOCO-seg and fine-tune on the derived masks; and implement the mask-conditioned Stage 2 with the masked reconstruction loss so an end-to-end comparison against the InstructPix2Pix baseline becomes possible."),
  P("Two limitations are acknowledged. First, because the ground-truth masks are recovered from the output of an instruction-following editor, Stage 1 learns where that editor makes changes and inherits its localization biases; validating a sample against human annotation is the planned mitigation. Second, the extraction threshold was tuned against simulated rather than real regeneration noise, and will be swept against a hand-labelled subset once generation completes.")
];

// ================= REFERENCES =================
const refs = [
  "[1]   T.-Y. Lin, M. Maire, S. Belongie, et al., “Microsoft COCO: Common Objects in Context,” ECCV, 2014.",
  "[2]   K. He, G. Gkioxari, P. Dollár and R. Girshick, “Mask R-CNN,” ICCV, 2017.",
  "[3]   D. Bolya, C. Zhou, F. Xiao and Y. J. Lee, “YOLACT: Real-time Instance Segmentation,” ICCV, 2019.",
  "[4]   X. Wang, R. Zhang, T. Kong, L. Li and C. Shen, “SOLOv2: Dynamic and Fast Instance Segmentation,” NeurIPS, 2020.",
  "[5]   B. Cheng, I. Misra, A. Schwing, A. Kirillov and R. Girdhar, “Masked-attention Mask Transformer for Universal Image Segmentation,” CVPR, 2022.",
  "[6]   G. Jocher, A. Chaurasia and J. Qiu, “Ultralytics YOLOv8,” 2023.",
  "[7]   T. Lüddecke and A. Ecker, “Image Segmentation Using Text and Image Prompts,” CVPR, 2022.",
  "[8]   Z. Yang, J. Wang, Y. Tang, et al., “LAVT: Language-Aware Vision Transformer for Referring Image Segmentation,” CVPR, 2022.",
  "[9]   A. Kirillov, E. Mintun, N. Ravi, et al., “Segment Anything,” ICCV, 2023.",
  "[10]  T. Ren, S. Liu, A. Zeng, et al., “Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks,” arXiv preprint, 2024.",
  "[11]  X. Lai, Z. Tian, Y. Chen, et al., “LISA: Reasoning Segmentation via Large Language Model,” CVPR, 2024.",
  "[12]  T. Brooks, A. Holynski and A. A. Efros, “InstructPix2Pix: Learning to Follow Image Editing Instructions,” CVPR, 2023.",
  "[13]  R. Rombach, A. Blattmann, D. Lorenz, P. Esser and B. Ommer, “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR, 2022.",
  "[14]  L. Zhang, A. Rao and M. Agrawala, “Adding Conditional Control to Text-to-Image Diffusion Models,” ICCV, 2023.",
  "[15]  G. Couairon, J. Verbeek, H. Schwenk and M. Cord, “DiffEdit: Diffusion-based Semantic Image Editing with Mask Guidance,” ICLR, 2023.",
  "[16]  A. Hertz, R. Mokady, J. Tenenbaum, et al., “Prompt-to-Prompt Image Editing with Cross-Attention Control,” ICLR, 2023.",
  "[17]  V. Singla, K. Yue, S. Paul, et al., “From Pixels to Prose: A Large Dataset of Dense Image Captions,” arXiv preprint, 2024.",
  "[18]  Gemma Team, “Gemma 3 Technical Report,” Google DeepMind, 2025.",
  "[19]  A. Radford, J. W. Kim, C. Hallacy, et al., “Learning Transferable Visual Models From Natural Language Supervision (CLIP),” ICML, 2021.",
  "[20]  A. Dosovitskiy, L. Beyer, A. Kolesnikov, et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR, 2021."
].map(t => new Paragraph({
  spacing: { after: 70, line: 260 }, indent: { left: 480, hanging: 480 },
  children: [new TextRun({ text: t, font: SERIF, size: 19 })]
}));

// ================= DOCUMENT =================
const doc = new Document({
  creator: "Amritha S; Yugeshwaran P",
  title: "Instruct-Seg-Edit — Review 1",
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 460, hanging: 240 } } }
      }]
    }]
  },
  sections: [{
    properties: { page: { margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } } },
    children: [
      ...title, ...abstract, ...intro, ...related, ...method, ...dataset, ...status,
      H1("REFERENCES"), ...refs
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("Instruct-Seg-Edit_Review1_Report.docx", buf);
  console.log("wrote Instruct-Seg-Edit_Review1_Report.docx");
});
