"""Template-derived teaching document; uses the bundled Python document runtime."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from PIL import Image,ImageDraw,ImageFont

S=Path(__file__).resolve().parent
OUT=S/'outputs'/'technical_guide';QA=OUT/'qa'
REF=Path('C:/Users/abdul/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-system-design/assets/reference.docx')
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
NS={'w':W,'m':M,'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}


def element(tag,text=None,**attrs):
    node=E.Element('{'+(M if tag.startswith('m:') else W)+'}'+tag.split(':')[-1])
    if text is not None:node.text=text
    for k,v in attrs.items():node.set('{'+W+'}'+k,str(v))
    return node


def mr(text):
    node=element('m:r');node.append(element('m:t',text));return node


def mathgroup(tag,*items):
    node=element('m:'+tag)
    for item in items:node.append(mr(item) if isinstance(item,str) else item)
    return node


def sub(base,index):return mathgroup('sSub',mathgroup('e',base),mathgroup('sub',index))
def sup(base,index):return mathgroup('sSup',mathgroup('e',base),mathgroup('sup',index))
def frac(num,den):return mathgroup('f',mathgroup('num',num),mathgroup('den',den))
def square_root(*items):
    rad=mathgroup('rad');props=mathgroup('radPr');hide=E.Element('{'+M+'}degHide');hide.set('{'+M+'}val','1')
    props.append(hide);rad.append(props);rad.append(mathgroup('deg'));rad.append(mathgroup('e',*items));return rad


class Guide:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True);QA.mkdir(exist_ok=True)
        with ZipFile(REF) as z:self.parts={n:z.read(n) for n in z.namelist()}
        self.tree=E.fromstring(self.parts['word/document.xml']);self.body=self.tree.find('w:body',NS)
        self.original=list(self.body);self.section=deepcopy(self.original[-1])
        self.para_template=deepcopy(self.original[24]);self.heading_template=deepcopy(self.original[23])
        self.table_template=deepcopy(self.original[40])
        self.picture_template=deepcopy(next(p for p in self.body.findall('w:p',NS) if p.find('.//w:drawing',NS) is not None))
        # The cover and metadata slots are retained; prose/table patterns are cloned below.
        for node in self.original[23:]:self.body.remove(node)
        cover_replacements={'System Name':'VISION HUNTERS','Title of Proposal':'Image Denoising Explained',
          '[Draft / Proposed / Approved]':'Technical guide','[Team Name]':'VISION_HUNTERS',
          '[Month, DD, YYYY]':'September 9, 2026','[Name(s)]':'VISION_HUNTERS project',
          'Reviewers':'Reader','[Reviewer names, roles, or groups]':'Basic digital image processing knowledge',
          '[Link to related docs]':'Implementation in scripts and recorded experiment results',
          '[One-sentence description of what this design covers]':'How the denoiser works and how we train and evaluate it'}
        for node in self.body.findall('.//w:t',NS):
            for old,new in cover_replacements.items():
                if node.text and old in node.text:node.text=node.text.replace(old,new)
        # Some template placeholders are split over multiple runs.
        for para in self.body.findall('.//w:p',NS):
            texts=para.findall('.//w:t',NS);joined=''.join(t.text or '' for t in texts)
            updated=joined
            for old,new in cover_replacements.items():updated=updated.replace(old,new)
            if updated!=joined and texts:
                texts[0].text=updated
                for t in texts[1:]:t.text=''
        for node in self.body.findall('w:p',NS):
            if node.find('w:pPr/w:pStyle[@w:val="Title"]',NS) is not None:self.black(node)
        evidence=dict(reference=str(REF),sha256=sha256(REF.read_bytes()).hexdigest(),pages=7,sections=1,
             parts={n:sha256(b).hexdigest() for n,b in self.parts.items()})
        (QA/'reference_inventory.json').write_text(json.dumps(evidence,indent=2))
        (QA/'artifact.md').write_text('''# Template contract
Reference and part SHA256 hashes are in reference_inventory.json. Seven reference pages were rendered with the packaged renderer using Word PDF export, then visually inspected.
One portrait US Letter section: 8.5 x 11 inches; top/left/right 0.7 inch; bottom 0.62014 inch; header/footer 0.5 inch; different first page. Preserve sectPr byte for byte.
Clone the centered cover, its two title paragraphs and metadata tables. Editable title/team/date/reader/scope slots are document.xml body children 8,9,20,22. Preserve page furniture and fonts.
Heading 1 pattern: 13.5 pt bold, after 6.5 pt, keep next/lines. Body pattern: Helvetica Neue, inherited size, 1.25 line spacing, after 5.5 pt, justified. Clone these patterns for educational sections. Black title/heading run overrides follow the document skill.
Clone the source dark navy table header, blue body shading, cell margins, repeat-header and no-split rules. Table content, row counts and column grids are editable according to the information presented. No fixed row heights.
Clone the source drawing paragraph and replace image1.png with a new pipeline diagram. Preserve the existing image relationship. Native OMML equations are new explanatory content.
Replace the generic section slots with twelve teaching sections in the same heading/body/table system. Unsupported service, tenant, reviewer and launch placeholders are removed. Add cloned section patterns as needed; deliberate page breaks begin teaching topics.
Footer organization and document title are editable. Footnote body is an editable definition slot; preserve the footnote part and its references when used. Preserve styles, numbering, fonts, settings, headers, relationships, section geometry and all other opaque parts byte for byte.
Content correctness and all final pages must be inspected. Final QA includes package comparison, absence of unfilled placeholders and retained reference checksum.
''',encoding='utf-8')

    def black(self,p):
        for color in p.findall('.//w:color',NS):color.set('{'+W+'}val','000000')
        for run in p.findall('w:r',NS):
            props=run.find('w:rPr',NS)
            if props is None:props=element('w:rPr');run.insert(0,props)
            color=props.find('w:color',NS)
            if color is None:color=element('w:color');props.append(color)
            color.set('{'+W+'}val','000000')

    def textnode(self,template,text):
        p=deepcopy(template);p.attrib.clear()
        props=p.find('w:pPr',NS)
        runprops=p.find('w:r/w:rPr',NS)
        for c in list(p):
            if c is not props:p.remove(c)
        if props is not None:
            for c in list(props):
                if E.QName(c).localname in ('numPr','sectPr'):props.remove(c)
        r=element('w:r')
        if runprops is not None:r.append(deepcopy(runprops))
        t=element('w:t',text);t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');r.append(t);p.append(r)
        return p

    def p(self,text):self.body.append(self.textnode(self.para_template,text))

    def heading(self,text,newpage=True):
        p=self.textnode(self.heading_template,text);self.black(p)
        props=p.find('w:pPr',NS)
        # Remove the source's footnote marker by constructing a fresh text run.
        if newpage:props.append(element('w:pageBreakBefore',val='1'))
        self.body.append(p)

    def equation(self,*items):
        p=self.textnode(self.para_template,'');p.find('w:pPr/w:jc',NS).set('{'+W+'}val','center')
        for r in p.findall('w:r',NS):p.remove(r)
        p.append(mathgroup('oMathPara',mathgroup('oMath',*items)));self.body.append(p)

    def table(self,rows,widths):
        table=deepcopy(self.table_template)
        source_rows=table.findall('w:tr',NS);head=deepcopy(source_rows[0]);body=deepcopy(source_rows[1])
        for row in source_rows:table.remove(row)
        grid=table.find('w:tblGrid',NS)
        for item in list(grid):grid.remove(item)
        for width in widths:grid.append(element('w:gridCol',w=round(width*1440)))
        table.find('w:tblPr/w:tblW',NS).set('{'+W+'}w',str(round(sum(widths)*1440)))
        for idx,values in enumerate(rows):
            row=deepcopy(head if idx==0 else body);cells=row.findall('w:tc',NS)
            celltemplate=deepcopy(cells[0])
            for cell in cells:row.remove(cell)
            for value,width in zip(values,widths):
                cell=deepcopy(celltemplate);props=cell.find('w:tcPr',NS)
                oldw=props.find('w:tcW',NS)
                if oldw is None:oldw=element('w:tcW',type='dxa');props.append(oldw)
                oldw.set('{'+W+'}w',str(round(width*1440)))
                p=cell.find('w:p',NS);newp=self.textnode(p,str(value))
                for oldp in cell.findall('w:p',NS):cell.remove(oldp)
                cell.append(newp);row.append(cell)
            table.append(row)
        self.body.append(table);self.p('')

    def diagram(self):
        image=Image.new('RGB',(2200,1100),'#f6f9fc');draw=ImageDraw.Draw(image)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',34);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
        bold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',43)
        draw.rectangle((0,0,2200,145),fill='#082a4a');draw.text((60,40),'The implemented image restoration pipeline',fill='white',font=bold)
        boxes=[(55,280,490,510,'Noisy image','RGB values in [0, 1]'),(590,280,1040,510,'Classical front end','Noise maps and wavelets'),
               (1140,280,1590,510,'Residual network','Nine input channels'),(1690,280,2140,510,'Protected output','Blend then save PNG')]
        for x,y,x2,y2,title,body in boxes:
            draw.rounded_rectangle((x,y,x2,y2),radius=9,fill='white',outline='#205b7c',width=3)
            draw.text((x+20,y+55),title,fill='#082a4a',font=font);draw.text((x+20,y+125),body,fill='#233447',font=small)
        for x in [490,1040,1590]:
            draw.line((x,395,x+90,395),fill='#2f6b9a',width=9);draw.polygon([(x+100,395),(x+72,377),(x+72,413)],fill='#2f6b9a')
        draw.line((265,515,265,680,1365,680,1365,515),fill='#2f6b9a',width=6)
        draw.text((490,695),'Original RGB is retained as an input',fill='#233447',font=font)
        draw.rounded_rectangle((590,820,1590,1000),radius=8,fill='#e4eff7',outline='#9fbfd7',width=2)
        draw.text((630,855),'Training only: clean targets, loss and optimizer',fill='#082a4a',font=font)
        draw.text((630,915),'Inference uses saved weights and the noisy image only',fill='#233447',font=small)
        path=QA/'pipeline.png';image.save(path);self.parts['word/media/image1.png']=path.read_bytes()
        p=deepcopy(self.picture_template);self.body.append(p)
        self.p('Figure 1. The classical estimate is the network output anchor. The noisy RGB and noise maps are also supplied to the network.')

    def save(self):
        self.body.append(self.section)
        self.parts['word/document.xml']=E.tostring(self.tree,xml_declaration=True,encoding='UTF-8',standalone=True)
        for name in ['word/footer1.xml','word/footer2.xml']:
            self.parts[name]=self.parts[name].replace(b'[Organization Name] | System Design RFC',b'VISION_HUNTERS | Image Denoising Explained')
        # No visible template footnote is retained in the teaching body; preserve the opaque footnote part.
        target=OUT/'VISION_HUNTERS_Technical_Guide.docx'
        with ZipFile(target,'w',ZIP_DEFLATED) as z:
            for name,data in self.parts.items():z.writestr(name,data)
        with ZipFile(REF) as source,ZipFile(target) as final:
            allowed={'word/document.xml','word/media/image1.png','word/footer1.xml','word/footer2.xml'}
            assert all(source.read(n)==final.read(n) for n in source.namelist() if n not in allowed)
        return target


def main():
    g=Guide()
    add_content(g)
    print(g.save())


def add_content(g):
    g.heading('1 How the solution works')
    g.p('This guide explains the VISION_HUNTERS implementation for a reader who knows basic filtering, image histograms and spatial or frequency representations. The main idea is to combine an adaptive classical denoiser with a network that learns how to correct its remaining errors. The same procedure is applied to every input image; it does not choose an algorithm by image filename or search for a matching clean photograph.')
    g.p('Your idea of identifying noise and applying a suitable filter is a useful starting point. Here, adaptation is mostly continuous: we estimate how strong the noise appears in different locations and color channels, then adjust the filtering strength. We do not first assign a hard label such as Gaussian or salt and pepper. Mixed noise, clipping and textured regions make a single label unreliable.')
    g.equation('y = x + n')
    g.p('In this simplified model, x is the unknown clean image, y is the observed noisy image and n is the error. Real corruption may be signal dependent, spatially varying or clipped, so this equation describes the residual without proving that n follows one distribution. Our task is to estimate x from y. During training, paired clean images teach the model what the correct answer should look like. During inference, no clean reference is available or required.')
    g.diagram()

    g.heading('2 Data and validation design')
    g.p('The public dataset contains 460 aligned noisy and clean RGB image pairs. Each supplied image is 992 by 992 pixels. The preliminary submission folder contains 20 noisy inputs with IDs 461 through 480 and no corresponding clean targets. The original files are preserved; caches, checkpoints and generated images are stored separately under scripts.')
    g.table([['Partition','Pairs','Purpose'],['Training','340','Update the network weights'],['Validation','60','Compare architectures, settings and checkpoints'],['Original held out test','60','One assessment after the earlier recipe was fixed'],['Preliminary inputs','20','Generate competition outputs; targets unavailable']], [1.8,.65,4.65])
    g.p('Splitting is done before cropping. Otherwise, patches from the same image could occur in both training and validation, making the score look better than true generalization. Visual review also identified 38 conservative groups of related views, and each group stays in one partition. These groups reduce obvious leakage but are not a guarantee that every location is independent.')
    g.p('The loader samples an image uniformly from the permitted image IDs, then selects a random crop. Despite the phrase scene uniform in a source comment, it does not sample the reviewed scene groups uniformly. A group with several images can therefore contribute more samples than a group with one image.')
    g.p('Important chronology: the earlier width 32 development model has already been assessed on the original held out partition, and a separate model has been trained on all 460 public pairs. The new width 48 and 64 experiments start from scratch using only the original 340 training images. Their selection uses the same 60 validation images; the old held out score must not be treated as a new independent test of these wider models.')
    g.p('Pixel arrays are converted from integers in [0, 255] to floating point in [0, 1]. This is numeric scaling, not gamma correction or exposure enhancement. Color and brightness should match the supplied clean target.')

    g.heading('3 Estimating local noise and repairing outliers')
    g.p('Implementation: classical.py, functions noise_maps and repair_outliers. The estimator divides the image into approximately 64 by 64 blocks and works separately on R, G and B. Inside each block it calculates a diagonal high frequency difference from every adjacent 2 by 2 group of pixels.')
    g.equation('h = ',frac('a − b − c + d','2'))
    g.p('Here a and b are the upper left and lower left values, while c and d are the upper right and lower right values. Constant intensity cancels. Under independent equal variance noise, the sum of squared filter coefficients is one, so this response has the same noise variance as the input. Edges and texture can still contribute to the response.')
    g.equation('σ = ',frac('median(|h − median(h)|)','0.67448975'))
    g.p('This is the median absolute deviation estimate. The divisor makes it consistent with the standard deviation of Gaussian noise under the model assumptions. A median is less sensitive to isolated extreme values than an ordinary average. However, it remains a heuristic estimate for clipped, correlated or structured corruption.')
    g.p('The coarse block estimates are smoothed with a Gaussian blur of standard deviation 0.8 in grid coordinates and resized to full resolution using bilinear interpolation. Estimates are clipped to [0, 0.3]. The result is a smoothly varying three-channel noise map, which allows stronger filtering in noisier regions without abrupt block boundaries.')
    g.p('Before wavelet filtering, the code makes a 3 by 3 median image. A pixel is pulled toward that median only when its absolute deviation exceeds max(4σ, 0.20), with a gradual blend above the threshold. The fixed 0.20 corresponds to roughly 51 intensity levels on the 8-bit scale. This is deliberately conservative because a bright detail or a thin branch can resemble an impulse. The original pixel values remain available to the network even if this classical branch changes them.')
    g.p('A practical interpretation: the noise map estimates where to be cautious; the median stage handles only extreme deviations. Neither step promises a perfect separation between noise and scene detail.')

    g.heading('4 Adaptive wavelet filtering')
    g.p('Implementation: classical.py, wavelet_estimate. The image is transformed into three orthonormal opponent channels: (R+G+B)/sqrt(3), (R−B)/sqrt(2), and (R−2G+B)/sqrt(6). The first describes common intensity and the other two describe color differences. This is an orthonormal basis, not the standard YCbCr transform.')
    g.p('For independent channel noise, the variance of each transformed channel is the sum of the RGB variances weighted by the squared transform coefficients. Correlated RGB noise violates this diagonal covariance approximation. The transform still separates shared structure from color variation, but its noise interpretation is then approximate.')
    g.p('Each opponent channel undergoes a Daubechies db2 wavelet decomposition with symmetric boundaries and up to three levels. The approximation coefficients are retained. Horizontal, vertical and diagonal detail coefficients are shrunk according to a local estimate of signal energy. Think of this as a multiscale filter bank: strong detail should survive, while weak coefficients dominated by noise should shrink.')
    g.equation(sub('σ','s'),' = ',square_root('max(E[c²] − ',sup(sub('σ','n'),'2'),', ε)'))
    g.equation('T = ',frac(sup(sub('σ','n'),'2'),sub('σ','s')),'     ',sub('c','out'),' = sign(c) max(|c| − T, 0)')
    g.p('For a detail coefficient c, E[c²] is estimated with a 5 by 5 average over squared coefficients. The resized noise map supplies the noise variance. Small numerical constants prevent division by zero. The implemented threshold has default strength 1.0. If estimated signal energy is strong, the threshold is lower; if a region appears mostly noisy, the threshold is higher.')
    g.p('For example, suppose a detail coefficient is 0.06, the noise variance is 0.0004 and estimated signal standard deviation is 0.02. The threshold is approximately 0.02, so soft thresholding produces 0.04. A coefficient of 0.01 under the same threshold becomes zero. Soft thresholding also reduces the magnitude of retained coefficients, which can attenuate fine texture.')
    g.p('Inverse wavelet reconstruction and the inverse opponent transform produce the classical RGB estimate B. This estimate is clipped to [0, 1] and quantized to 8-bit precision for consistency with the training cache. The wavelet method already improves the validation score on its own. The network learns additional corrections rather than being responsible for all noise removal from the start.')

    g.heading('5 The residual neural network')
    g.p('Implementation: model.py. The network input is a concatenation of the noisy RGB image, the classical RGB estimate, and the three noise maps. Nine channels do not mean nine colors: they are nine aligned numeric feature planes describing the same scene.')
    g.equation('z = concat(y, B, σ)     ',sub('x','raw'),' = B + ',sub('f','θ'),'(z)')
    g.p('The learned function f with parameters θ predicts a signed correction to the classical estimate. It can recover details removed by filtering or subtract noise the filter left behind. The original noisy RGB is retained so the model can inspect evidence that the classical result may have lost. The final convolution starts at zero, so an untrained hybrid initially outputs the classical estimate.')
    g.table([['Stage for a 256 pixel crop','Spatial size','Channels','NAF blocks'],['Encoder 1','256 × 256','w','1'],['Encoder 2','128 × 128','2w','1'],['Encoder 3','64 × 64','4w','2'],['Encoder 4','32 × 32','8w','4'],['Middle','16 × 16','16w','4'],['Decoder stages','32 → 256','8w → w','1 at each scale']], [2.65,1.45,1.1,1.9])
    g.p('A stride-2 convolution halves the spatial dimensions and doubles channels. The decoder upsamples using a 1 by 1 convolution followed by PixelShuffle. Additive skip connections transfer encoder features at the same scale to the decoder. This combines broad context at low resolution with precise spatial detail at high resolution. Inputs are padded to a multiple of 16 and cropped back afterward.')
    g.p('Each NAF block normalizes across channels at each pixel, expands channels with a 1 by 1 convolution, applies a depthwise 3 by 3 convolution, then splits the features into two halves and multiplies them. This SimpleGate is nonlinear even though there is no ReLU. Global average pooling supplies a learned channel attention signal. A second gated channel-mixing branch follows. Learned residual scales start at zero to make the blocks initially close to identity.')
    g.p('Width w is the number of channels at the first scale. Width 32 therefore reaches 512 channels in the middle; width 64 reaches 1024. Width is neither the image size nor the number of training steps. Wider networks have more capacity and usually higher memory and runtime cost. They may still overfit or learn less useful corrections, so their value must be measured.')

    g.heading('6 What happens in a training step')
    g.p('Implementation: prepare_cache.py, training_data.py and train.py. Real paired examples use cached noisy RGB, quantized classical RGB, quantized noise maps and clean RGB. This is a 12-channel uint8 cache. Noise maps are encoded over [0, 0.3] and decoded before use. Hashes of the preprocessing code and data split guard against silently training on stale features.')
    g.p('For the standard recipe, approximately 70% of sampled crops use supplied noisy and clean pairs. Another 20% create fresh corruption from a clean training crop, and 10% are identity examples whose input equals the clean target. Identity examples teach the network that a clean input should need little correction. The real-pairs-only ablation disables both synthetic and identity branches.')
    g.p('Synthetic corruption combines a small Gaussian read component, intensity-dependent variance, a horizontal variation in noise strength, and sometimes sparse extreme-valued samples. The image is clipped and quantized afterward. These examples broaden the training distribution; they do not reconstruct the organizer noise generator. Rotations by multiples of 90 degrees and horizontal flips are applied consistently to inputs and targets.')
    g.p('The first 1,000 training updates use 128 by 128 crops; subsequent updates use 256 by 256 crops. In this implementation, warmup_steps controls crop size, not a learning-rate warmup. Every optimizer update uses an effective batch of 16 crops. The usual microbatch is 8 with two accumulated backward passes. A microbatch of 4 with four passes uses the same effective batch while reducing peak activation memory. Floating-point summation order can still change slightly.')
    g.equation(sub('L','MSE'),' = ',frac('1','N'),' ∑ (',sub('x','pred'),' − ',sub('x','target'),')²')
    g.p('MSE averages squared pixel errors over the batch, channels and crop dimensions. Backpropagation calculates how each weight affected that error. AdamW updates the weights with an initial learning rate of 0.0002, weight decay 0.0001 and a cosine learning-rate schedule defined over 30,000 steps. Runs stopped at 20,000 therefore stop before that schedule reaches its minimum. Gradients are clipped to norm 1.')
    g.p('GPU training uses mixed precision with gradient scaling; normalization variance and loss calculations retain float32 precision. An exponential moving average of the weights smooths update noise, and those EMA weights are evaluated and exported. Loss is a training signal; the saved best model is chosen by the official validation score. A falling loss does not guarantee a rising validation score.')
    add_remaining(g)


def add_remaining(g):
    g.heading('7 Evaluating the competition score')
    g.p('Implementation: metrics.py calls the unchanged official evaluation/evaluate.py. Every validation image is denoised at full resolution, then the prediction is clipped and rounded as an actual PNG would be. Scoring the saved representation matters because float output and 8-bit output can differ slightly. Validation is performed every 1,000 updates and at the end of a development run.')
    g.equation('PSNR = 10 ',sub('log','10'),'(',frac('1','MSE'),')')
    g.p('PSNR measures pixel accuracy. This expression assumes images normalized to [0, 1]. If MSE is halved, PSNR increases by about 3 dB. SSIM measures local similarity using means, variances and covariance; the official implementation uses a uniform 7 by 7 window, sample covariance and data range 1. SSIM complements MSE but is not a perfect measure of visual quality.')
    g.equation(sub('Q','i'),' = 0.6 clip(',frac('ΔPSNRᵢ','15'),', 0, 1) + 0.4 max(ΔSSIMᵢ, 0)')
    g.p('For each image, the deltas are improvements over that same noisy input. The final score is the average of the per-image Q values. The clipping and maximum must be applied before averaging. A method cannot be ranked correctly just by inserting average deltas into this nonlinear formula.')
    g.p('Example: an input has PSNR 20 dB and SSIM 0.45. Its output reaches 29 dB and SSIM 0.85. The improvements are 9 dB and 0.40, so the image score is 0.6 × (9/15) + 0.4 × 0.40 = 0.52. This is an explanatory example, not a measured image from the dataset.')
    g.p('Initial training minimizes MSE. Later experiments start from the best MSE checkpoint and optimize MSE + λ(1 − SSIM), with λ equal to 0.05 or 0.10. Each fine-tune has its own 2,000 updates, learning rate 0.00005 and 2,000-step cosine schedule. The same full validation score chooses the winner; the loss weights are not the competition scoring weights.')
    g.p('Validation is used repeatedly for model selection, so its best observed score can be optimistic. The earlier one-time held out assessment provides separate evidence for the earlier selected development checkpoint. It does not validate every subsequent architecture or the model retrained on all public pairs.')

    g.heading('8 Inference and protection for mild noise')
    g.p('Implementation: inference.py and denoise.py. Inference loads one local checkpoint, switches the model to evaluation mode and disables gradients. It estimates features from the supplied image and predicts corrections. It runs in float32 on either CPU or CUDA; it does not train again for each user photograph and does not require internet access.')
    g.p('The neural stage processes 512 by 512 tiles with a nominal 64-pixel overlap. The final tile position is adjusted to cover the image boundary, so actual overlap can be larger. Each prediction is multiplied by a positive Hann window and overlapping values are averaged using their accumulated weights. This reduces tile seams and limits GPU activation memory. The full feature maps and output arrays still occupy host memory, so arbitrary dimensions do not mean unlimited image size.')
    g.p('Global channel attention sees each tile rather than the whole photograph. Consequently, tiled and full-frame results need not be mathematically identical. A prototype check found a very small PSNR difference on one tested image, but this is not proof of equivalence for all models and images.')
    g.equation(sub('x','out'),' = y + α(',sub('x','raw'),' − y),     0 ≤ α ≤ 1')
    g.p('The optional mild-noise protection calculates a blend weight α. When there is little noise evidence, the output stays close to the input. When evidence is strong, the neural correction is used more fully. This addresses a failure seen in an early prototype: a denoiser trained mainly on heavily corrupted images could unnecessarily modify an already clean photograph.')
    g.p('Noise evidence comes from two sources. First, noise maps are computed in the two opponent chroma channels and averaged. Their local confidence is clip(chroma_sigma/threshold, 0, 1) squared. Second, the mean RGB intensity is divided into sampled 7 by 7 patches. The weakest-texture quarter is used to form a covariance matrix; the mean of its eight lowest nonnegative eigenvalues estimates a luminance noise variance. This helps detect noise shared across RGB channels or grayscale noise that chroma alone could miss.')
    g.p('The luminance confidence is clip((luma_sigma − 0.02)/0.03, 0, 1) squared, and the larger of the two confidences is used. This combines a local chroma map with a global luminance estimate. Protection thresholds are validated rather than assumed universal. A threshold of zero disables protection; the earlier selected width 32 checkpoint uses 0.02.')
    g.p('Finally, values are clipped and rounded to an 8-bit PNG. Grayscale inputs are converted to RGB internally; any alpha channel is preserved separately. No automatic brightness or contrast enhancement is applied.')

    g.heading('9 Experiments and model width')
    g.p('The following completed results use the same 60 validation images. Network rows show the saved 4,000-step evaluations without the later protection calibration. These are local development measurements, not official leaderboard scores.')
    rows=[['Method','PSNR dB','SSIM','Composite']]
    sources=[('Starter baseline',S/'runs'/'baseline_val'/'metrics.json'),('Adaptive wavelet',S/'runs'/'wavelet_val_1'/'metrics.json'),
             ('RGB only width 16',S/'runs'/'rgb16'/'val_004000.json'),('Hybrid width 16',S/'runs'/'hybrid16'/'val_004000.json'),
             ('Hybrid width 32',S/'runs'/'hybrid32'/'val_004000.json'),('Hybrid 16 real pairs only',S/'runs'/'hybrid16_real_only'/'val_004000.json')]
    for label,path in sources:
        m=json.loads(path.read_text())['summary'];rows.append([label,f'{m["psnr"]:.2f}',f'{m["ssim"]:.4f}',f'{m["composite_score"]:.5f}'])
    g.table(rows,[3.45,1.2,1.1,1.35])
    g.p('The width 16 hybrid beats the RGB-only control at equal update count, supporting the usefulness of the classical inputs and anchor in this recipe. It does not isolate each classical operation individually. Width 32 improves further, suggesting the smaller model was capacity-limited or learned more slowly at this budget. Equal steps are not equal wall time or equal compute.')
    g.p('After extension to 20,000 updates, width 32 reached a best validation score of about 0.60438. Its selected SSIM fine-tune reached about 0.60698 after protection calibration. That selected development model scored 0.64765 on the original held out partition. The test partition contains different images, so the larger test number does not imply an extra improvement from the same model on the validation set.')
    g.p('A separate width 32 model has since completed the fixed recipe on all 460 public pairs, and a local 20-image candidate ZIP was produced. Its weights are different from the development checkpoint. The earlier held out score must not be attributed to the all-data model, because those images were part of its training.')
    g.p('The new experiment plan trains widths 48 and 64 to 4,000 steps first. Each model whose exact step-4,000 score strictly exceeds 0.5646054497 is extended to 20,000 steps. Both can qualify. A memory probe chooses a microbatch while keeping the effective batch at 16. Width 48 has passed the probe; results for the new widths are still pending in this guide. CPU timing and robustness checks follow, and the earlier candidate remains preserved.')

    g.heading('10 Running the software and reading progress')
    g.p('All custom implementation files are under scripts. The required denoising command accepts noise_dir and denoised_dir. A competition name such as 461_noise.png becomes 461.png. Ordinary input names retain their stem. Name collisions, missing weights and overlapping input/output directories produce explicit errors. Existing output images require an explicit overwrite option.')
    g.p('From the project folder, a concrete CPU invocation for the preserved candidate is shown below. The PowerShell backtick continues a command onto the next line.')
    for line in ['.\\scripts\\.venv\\Scripts\\python.exe scripts/denoise.py `',
                 '  --noise_dir submissions/noisy `',
                 '  --denoised_dir scripts/outputs/manual_demo `',
                 '  --checkpoint scripts/checkpoints/VISION_HUNTERS_final_candidate.pt `',
                 '  --device cpu']:
        g.p(line)
    g.p('Double-click scripts/Watch Training.cmd to open the live progress monitor. A run marked complete only means that particular model reached its requested stopping point. The full queue can still include another width, longer training, evaluation or packaging. Wait for ALL QUEUED WORK FINISHED before turning off the laptop. The monitor is read-only; closing it does not stop the separate training process.')
    g.table([['Displayed item','Meaning'],['Width','Number of channels at the first network scale'],['Step','One optimizer update using an effective batch of 16'],['Loss','Training objective on recent sampled crops'],['Best val score','Highest recorded score on the validation partition'],['Validating','Full-image scoring is running; step counter temporarily pauses'],['GPU peak MB','Maximum PyTorch tensor allocation observed in that process']], [2.05,5.05])
    g.p('Training saves the optimizer, gradient scaler, current weights, EMA weights and step in resumable checkpoints, usually every 250 updates. Logs and source hashes support diagnosis and provenance. This supports resuming interrupted work, but does not promise bitwise-identical training after every interruption because not every random-generator state is restored. Deterministic inference is checked separately.')

    g.heading('11 Generalization and technical limitations')
    g.p('Supporting many image dimensions and file types is different from restoring every possible corruption well. The strongest evidence currently comes from the supplied photograph distribution. Robustness diagnostics use eight fixed validation crops under clean, mild and strong Gaussian, signal-dependent, spatially varying, grayscale and correlated noise conditions. They test useful failure cases, but they are not a substitute for unrelated cameras and scenes.')
    g.p('Texture can be mistaken for noise. Wavelet shrinkage can attenuate fine detail, and learned correction can introduce smoothing or artifacts when the test image differs from training. The mild-noise blend can help preserve clean content, but its confidence is statistical rather than a perfect detector. It can also reduce useful denoising when noise is underestimated.')
    g.p('Clipping discards information: if several clean intensities become the same saturated value, exact recovery is ambiguous. A learned image prior can make a plausible estimate, but it cannot guarantee the original pixel value. The current synthetic augmentation also does not explicitly model every JPEG artifact, demosaicing pattern, stripe, blur or compression failure.')
    g.p('Increasing width adds parameters and intermediate features, not new training information. Width 48 contains 25,944,819 trainable parameters, compared with 2,927,187 for width 16. Many channel-mixing weights grow roughly with the square of width. Peak memory also includes activations, gradients, optimizer states and EMA weights, so it cannot be inferred from parameter count alone. Measured GPU memory and CPU runtime decide whether a larger model is practical.')
    g.p('Repeatedly trying models on the same validation set increases the chance of selecting a model that happens to fit those images unusually well. The score should be read alongside individual-image behavior, worst-performing cases, speed and robustness. A wider model exceeding the 4,000-step threshold earns a longer experiment; it does not automatically become the final submission.')
    g.p('The original held out set has already served its one-time role for the previous recipe. If later changes are informed by that result, it cannot be described as untouched evidence for those changes. The new width queue therefore uses the existing training and validation partitions and leaves the earlier assessment and candidate files unchanged.')

    g.heading('12 Code map and a suggested reading order')
    g.table([['File under scripts','What to read for'],['classical.py','Noise maps, outlier repair, opponent transform and wavelet thresholds'],['model.py','NAF blocks, encoder/decoder, residual anchor and channel width'],['training_data.py','Crop selection, augmentation and real/synthetic/identity mixture'],['train.py','Loss, optimizer, EMA, checkpoints and validation schedule'],['metrics.py','Connection to the official evaluator and score aggregation'],['inference.py','Local checkpoint loading, tiled prediction and protection blend'],['denoise.py','User command, image handling, names and output checks'],['run_experiments.py','Earlier comparisons, extension and SSIM selection'],['run_width_experiments.py','Width 48/64 comparison and conditional 20,000-step extension'],['prepare_final_candidate.py','Earlier fixed-recipe assessment and all-data preparation'],['package_submission.py and status.py','ZIP integrity and live progress reporting']], [2.55,4.55])
    g.p('Start with classical.py and trace a single image through noise_maps, wavelet_estimate and features. Next read Denoiser.forward in model.py: identify the nine input channels, the classical anchor and the added correction. Then follow predict in inference.py to see how that model is applied to a full photograph. Read train.py last to connect the forward pass with a loss and weight updates.')
    g.p('Useful terms: a tensor is a multidimensional numeric array; a feature map is one spatial channel inside the network; a residual is a predicted correction; a checkpoint stores model state; an ablation removes or changes one component to measure its contribution; EMA averages weights over updates. The RGB-only control and real-pairs-only model are ablation experiments.')
    g.p('Source of implementation details: the local Python files above, scripts/data/split.json, recorded run configurations and validation JSON files. Architecture attribution: Chen et al., Simple Baselines for Image Restoration, ECCV 2022, with the NAFNet reference at https://github.com/megvii-research/NAFNet. The network is implemented with standard PyTorch operations and trained from scratch; no external pretrained weights were used.')
    g.p('Competition definitions come from the Participant Handbook and the unchanged starter evaluator at https://github.com/moraspcup-official/mora_sp_cup_2026, source commit 6fe15af99b2dc5d808cf6004f7cb11df148b9f82. Metric records and checkpoint hashes in the workspace identify the measurements described here. The guide is a technical explanation, separate from the shorter competition report.')


if __name__=='__main__':main()
