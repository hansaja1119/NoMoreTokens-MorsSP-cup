"""Four-page development report. Does not claim a frozen competition submission.

Uses reportlab from the bundled document runtime; metrics come from saved runs.
"""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.utils import ImageReader
from PIL import Image

S=Path(__file__).resolve().parent;ROOT=S.parent
OUT=S/'outputs'/'development'/'NoMoreTokens_Report.pdf'
W,H=A4;M=72;CW=W-2*M
INK=colors.HexColor('#172635');BLUE=colors.HexColor('#205b7c')
STYLE=ParagraphStyle('body',fontName='Helvetica',fontSize=12,leading=15.5,textColor=INK,spaceAfter=8)
SMALL=ParagraphStyle('table',parent=STYLE,fontSize=12,leading=14)

def read(path):return json.loads(path.read_text())

class Report:
    def __init__(self):
        OUT.parent.mkdir(parents=True,exist_ok=True)
        self.c=canvas.Canvas(str(OUT),pagesize=A4)
        self.c.setTitle('NoMoreTokens - Adaptive Denoising Development Report')
        self.c.setAuthor('NoMoreTokens');self.page=0
    def start(self,title):
        if self.page:self.finish()
        self.page+=1;self.y=H-M-12
        self.c.setFillColor(BLUE);self.c.setFont('Helvetica-Bold',12)
        self.c.drawString(M,self.y,'NoMoreTokens  |  MORA SP CUP 2026');self.y-=31
        self.c.setFillColor(INK);self.c.setFont('Helvetica-Bold',18)
        self.c.drawString(M,self.y,title);self.y-=27
    def p(self,text):
        para=Paragraph(text,STYLE);_,height=para.wrap(CW,self.y-M)
        if self.y-height<M+35:raise RuntimeError(f'Page {self.page} overflow: {text[:60]}')
        para.drawOn(self.c,M,self.y-height);self.y-=height+9
    def heading(self,text):
        self.y-=4;self.c.setFont('Helvetica-Bold',12);self.c.setFillColor(BLUE)
        self.c.drawString(M,self.y,text);self.y-=20
    def table(self,rows,widths):
        table=Table([[Paragraph(escape(str(cell)),SMALL) for cell in row] for row in rows],colWidths=widths)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff4')),
          ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),
          ('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,BLUE),
          ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#ccd5dc'))]))
        _,height=table.wrap(CW,600)
        if self.y-height<M+35:raise RuntimeError(f'Page {self.page} table overflow')
        table.drawOn(self.c,M,self.y-height);self.y-=height+12
    def finish(self):
        self.c.setStrokeColor(colors.HexColor('#ccd5dc'));self.c.line(M,94,W-M,94)
        self.c.setFont('Helvetica',12);self.c.setFillColor(INK)
        self.c.drawString(M,76,'Development evidence - not a frozen submission')
        self.c.drawRightString(W-M,76,str(self.page));self.c.showPage()
    def close(self):self.finish();self.c.save()

def main():
    baseline=read(S/'runs'/'baseline_val'/'metrics.json')['summary']
    wavelet=read(S/'runs'/'wavelet_val_1'/'metrics.json')['summary']
    rgb_path=S/'runs'/'rgb16'/'val_004000.json'
    rgb=read(rgb_path)['summary']
    h1=read(S/'runs'/'hybrid16'/'val_001000.json')['summary']
    h4=read(S/'runs'/'hybrid16'/'val_004000.json')['summary']
    stress=read(S/'runs'/'hybrid16'/'robustness_1000_mixed_guard.json')['summary']
    runtime=read(S/'runs'/'hybrid16'/'runtime_1000.json')
    bm3d=read(S/'runs'/'bm3d_val_subset'/'metrics.json')
    report=Report()
    report.start('Adaptive image denoising')
    report.p('<b>Status: September 8, 2026.</b> This report records completed development experiments. '
             'Architecture comparisons and longer training are still running. The locked test and '
             'organizer-held targets have not been used to select the model.')
    report.heading('Problem and objective')
    report.p('Recover the supplied clean photograph from a noisy RGB image while preserving detail, '
             'color and exposure. The submission must also run locally and offline on a CPU. We '
             'combine input-adaptive signal processing with a compact supervised restoration network.')
    report.p('The handbook allocates 40% to the report and analysis, 10% to significant classical '
             'processing, 35% to hidden-image quality, and 15% to code, compliance and runtime. '
             'Our objective therefore includes clear ablations and reproducibility, alongside quality.')
    report.heading('Dataset observations')
    report.table([['Property','Verified observation'],['Public data','460 noisy / clean pairs'],
       ['Preliminary inputs','20 noisy images, IDs 461-480'],['Image format','All 940 files: 992 x 992 RGB PNG'],
       ['Raw noisy PSNR','15.45 to 30.17 dB; median 19.14 dB'],
       ['Observed content','Foliage, trees, walls, buildings and walkways']], [CW*.34,CW*.66])
    report.p('Residuals show heterogeneous severity, clipping and large deviations. This does not '
             'identify a unique physical sensor model. Noise parameters used for training are estimated '
             'from training pairs only. Exposure enhancement is excluded because the target is the '
             'provided clean image, not an artistically brightened result.')
    report.y-=12
    box=(CW-32)/3
    for n,label in enumerate(['Noise analysis','Wavelet estimate','Learned correction']):
        x=M+n*(box+16)
        report.c.setFillColor(colors.HexColor('#e7eff4'))
        report.c.roundRect(x,report.y-38,box,38,5,fill=1,stroke=0)
        report.c.setFillColor(INK);report.c.setFont('Helvetica',12)
        report.c.drawCentredString(x+box/2,report.y-23,label)
        if n<2:
            report.c.setStrokeColor(BLUE);report.c.line(x+box+2,report.y-19,x+box+14,report.y-19)
            report.c.line(x+box+10,report.y-16,x+box+14,report.y-19)
            report.c.line(x+box+10,report.y-22,x+box+14,report.y-19)
    report.y-=52
    report.p('The original RGB image and estimated noise maps also enter the network.')

    report.start('Method and validation design')
    report.heading('A single adaptive procedure for every input')
    report.p('<b>1. Estimate noise:</b> robust diagonal high-frequency differences produce smooth, '
             'spatially varying RGB noise-strength maps. No image ID or clean-reference lookup is used.')
    report.p('<b>2. Classical estimate:</b> cautiously repair strong isolated deviations, transform '
             'to orthonormal opponent colors, then apply multiscale wavelet shrinkage with thresholds '
             'adapted to estimated noise and local coefficient energy.')
    report.p('<b>3. Learned correction:</b> a four-scale NAF-block network receives nine channels: '
             'original RGB, classical RGB and three noise maps. It predicts a correction to the '
             'classical estimate while retaining the original detail as an input. The width-16 '
             'model has 2,927,187 parameters. The RGB-only control starts from the noisy image.')
    report.p('<b>4. Mild-noise protection:</b> an experimental continuous blend reduces corrections '
             'when opponent-chroma noise is weak. Low-eigenvalue variance from weak-texture luminance '
             'patches supplies separate evidence for correlated and grayscale noise. Thresholds are '
             'calibrated on validation data and must be retested for each final candidate.')
    report.heading('Training and leakage control')
    report.p('A fixed split contains 340 training, 60 validation and 60 locked-test images. Visual '
             'review identified 38 conservative groups of related views; every group stays within '
             'one partition. Crops are sampled only after splitting. Grouping is heuristic and does '
             'not establish independence across all physical locations.')
    report.p('Training uses matched rotations/flips, 128-pixel crops followed by 256-pixel crops, '
             'AdamW, mixed precision, exponential moving-average weights and an effective batch of 16. '
             'The initial mixture is 70% supplied pairs, 20% fresh synthetic corruptions and 10% identity '
             'pairs. Initial models use MSE; SSIM fine-tuning and a real-pairs-only ablation are pending.')
    report.p('Full validation uses PNG-quantized outputs and the unchanged official evaluator. '
             'The composite is calculated per image and then averaged: '
             '<b>Q = 0.6 clip(delta PSNR / 15, 0, 1) + 0.4 max(delta SSIM, 0).</b> '
             'Missing images are rejected by our benchmark workflow.')

    report.start('Measured development results')
    rows=[['Method','Steps','PSNR','SSIM','Score']]
    for label,steps,m in [('Starter baseline','-',baseline),('Adaptive wavelet','-',wavelet),
                           ('RGB-only network','4,000',rgb),('Hybrid network','4,000',h4)]:
        rows.append([label,steps,f'{m["psnr"]:.2f}',f'{m["ssim"]:.4f}',f'{m["composite_score"]:.4f}'])
    report.table(rows,[CW*.37,CW*.15,CW*.16,CW*.16,CW*.16])
    report.p('All rows use the same 60 validation images. These are development results, not '
             'organizer scores. The network rows above do not include mild-noise protection. '
             'The equal-step RGB/hybrid comparison measures learning progress, not final converged capacity.')
    report.p(f'The 4,000-step hybrid gains <b>{h4["composite_score"]-baseline["composite_score"]:.4f}</b> '
             'over the starter (95% paired scene-bootstrap interval [0.2369, 0.3209], 41 groups). '
             f'Its worst-decile mean score is {h4["worst_decile_composite"]:.4f}.')
    report.p(f'On eight severity-spanning validation images, BM3D scored {bm3d["summary"]["composite_score"]:.4f}, '
             f'versus {bm3d["same_image_comparisons"]["hybrid16_4000"]["composite_score"]:.4f} for the hybrid. '
             f'BM3D took a median {bm3d["summary"]["median_seconds"]:.1f} seconds per image on four CPU threads.')
    report.heading('Fixed central crops: clean / noisy / wavelet / hybrid')
    sample_ids=read(S/'runs'/'hybrid16_cpu_4000'/'metrics.json')['rows'][:2]
    cell=94
    for row in sample_ids:
        id_=row['id'];paths=[ROOT/'public'/'ground_truth'/f'{id_}.png',ROOT/'public'/'noisy'/f'{id_}_noise.png',
                            S/'runs'/'wavelet_val_1'/'images'/f'{id_}.png',S/'runs'/'hybrid16_cpu_4000'/'images'/f'{id_}.png']
        for n,path in enumerate(paths):
            with Image.open(path) as im:crop=im.crop((368,368,624,624)).copy()
            report.c.drawImage(ImageReader(crop),M+n*((CW-cell)/3),report.y-cell,cell,cell)
        report.y-=cell+5
        report.c.setFont('Helvetica',12);report.c.drawString(M,report.y-12,f'Image {id_}; hybrid crop includes the experimental protection.')
        report.y-=30
    if report.y<M+35:raise RuntimeError('Crop layout overflow')

    report.start('Robustness, engineering and next steps')
    report.heading('Broader-noise diagnostics')
    report.p('Eight fixed validation crops were tested under clean, mild/strong Gaussian, '
             'signal-dependent, spatial-mixture, grayscale and correlated-noise conditions. '
             'These synthetic diagnostics are narrower evidence than testing unrelated cameras.')
    report.p('On the 1,000-step prototype, the combined protection preserved all eight checked '
             'clean color and grayscale crops exactly after PNG rounding. Mild-noise mean squared '
             f'error fell from {stress["mild_gaussian"]["input_mse"]:.6f} to '
             f'{stress["mild_gaussian"]["output_mse"]:.6f}. Strong and correlated-noise cases also '
             'improved over their inputs. This is not a guarantee for every photograph or corruption.')
    report.heading('Offline runtime and reproducibility')
    report.p('Inference supports the required directory flags, automatic suffix removal, arbitrary '
             'dimensions, RGB/grayscale images, alpha preservation and explicit CPU fallback. It loads '
             'one local checkpoint and uses float32 inference with 512-pixel tiles and 64-pixel overlap.')
    report.p(f'For one prototype image, CPU/GPU outputs differed by at most one 8-bit level; '
             f'tiled/full-frame PSNR differed by {abs(runtime["tiled_psnr"]-runtime["full_frame_psnr"]):.5f} dB. '
             'Fourteen tests pass, covering CLI behavior, metric parity, selection gates and final-training isolation.')
    report.p('A clean CPU-only installation reproduced all 20 prototype PNGs byte-for-byte in '
             '69.7 seconds, versus 66.6 seconds for CPU inference in the CUDA-capable environment. '
             'Final-checkpoint runtime checks remain pending.')
    report.heading('Remaining work before submission')
    report.p('Finish architecture comparisons, fine-tuning and robustness selection; assess the '
             'locked test once, then retrain the fixed recipe on all public pairs. Reproduce the '
             '20 outputs, finalize this report, arrange private-repository access and freeze the submission.')
    report.heading('Sources and attribution')
    report.p('Participant Handbook, Mora SP Cup 2026. '
             '<link href="https://github.com/moraspcup-official/mora_sp_cup_2026">Official starter repository</link>, '
             'commit 6fe15af99b2dc5d808cf6004f7cb11df148b9f82. '
             'Chen et al., <i>Simple Baselines for Image Restoration</i>, ECCV 2022; '
             '<link href="https://github.com/megvii-research/NAFNet">NAFNet architecture reference</link>. '
             'No external training data or pretrained weights were used.')
    report.close();print(OUT)

if __name__=='__main__':main()
