"""Build a research manuscript with Methods, Extended Data and source-derived tables."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common.runtime import ROOT,save_json


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


from src.common.report_typesetting import MM as mm


MODEL_NAMES = {'cls_r50_ce': 'Cross-entropy', 'cls_r50_wce': 'Weighted cross-entropy',
               'cls_r50_focal': 'Focal loss', 'seg_unet3d': '3D U-Net', 'seg_unet2d': '2D U-Net'}
REFERENCES = {
    15: 'ISIC. ISIC 2018 Task 3 data access and attribution. https://challenge.isic-archive.com/data/ (accessed 11 October 2026).',
    16: 'Creative Commons. Attribution–NonCommercial 4.0 International licence. https://creativecommons.org/licenses/by-nc/4.0/ (accessed 11 October 2026).',
    1: 'Codella, N. et al. Skin lesion analysis toward melanoma detection 2018. arXiv:1902.03368 (2019).',
    2: 'Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data 5, 180161 (2018).',
    3: 'He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition. Proc. CVPR, 770–778 (2016).',
    4: 'Lin, T.-Y. et al. Focal loss for dense object detection. Proc. ICCV, 2999–3007 (2017).',
    5: 'Ronneberger, O., Fischer, P. & Brox, T. U-Net: convolutional networks for biomedical image segmentation. Proc. MICCAI, 234–241 (2015).',
    6: 'Çiçek, Ö. et al. 3D U-Net: learning dense volumetric segmentation from sparse annotation. Proc. MICCAI, 424–432 (2016).',
    7: 'Xiong, Z. et al. A global benchmark of algorithms for segmenting the left atrium from late gadolinium-enhanced cardiac magnetic resonance imaging. Medical Image Analysis 67, 101832 (2021).',
    8: 'MedPy. Binary segmentation metrics: assd and hd95. https://loli.github.io/medpy/generated/medpy.metric.binary.assd.html (accessed 11 October 2026).',
    9: 'Selvaraju, R. R. et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. Proc. ICCV, 618–626 (2017).',
    10: 'Yu, L. et al. UA-MT left atrial preprocessing code, commit 88ed29ad794f877122e542a7fa9505a76fa83515. https://github.com/yulequan/UA-MT (accessed 11 October 2026).',
    11: 'pynrrd. Reading API and array index order. https://pynrrd.readthedocs.io/en/stable/reference/reading.html (accessed 11 October 2026).',
    14: 'Kassis, T., Agarwal, V., He, Y., Patel, D. & Brueckner, A. M. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065 (2026). https://doi.org/10.48550/arXiv.2609.00065.',
    13: 'Tzeis, S. et al. 2024 EHRA/HRS/APHRS/LAHRS expert consensus statement on catheter and surgical ablation of atrial fibrillation. Europace 26, euae043 (2024).',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def percent(value):
    return f'{100 * float(value):.2f}'



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    parser.add_argument('--output', default='report/report.pdf')
    args = parser.parse_args()
    results, output = ROOT/args.results, ROOT/args.output
    analysis = read_json(ROOT/'report/analysis.json')
    editorial = read_json(ROOT/'report/editorial_content.json')
    if len(analysis['anatomical_interpretation']) != 3 or len(analysis['gradcam']) != 4 or len(analysis['segmentation_cases']) != 4:
        raise ValueError('The anatomical discussion and all visual cases must be complete')
    cls = read_csv(results/'classification/summary.csv')
    seg = read_csv(results/'segmentation/summary.csv')
    if len(cls) != 3 or len(seg) != 2:
        raise ValueError('All five original models must be reported')
    from src.common.report_typesetting import LatexManuscript
    doc = LatexManuscript(results, ROOT, REFERENCES)
    doc.para(editorial['title'], 'Title')
    doc.para('MedVision Bench · Controlled comparisons in medical image analysis', 'Meta')
    doc.para(editorial['abstract'], 'Summary')
    for paragraph in editorial['introduction']:
        doc.para(paragraph)
    diagram = read_json(results/'figures/method_diagram_provenance.json')
    doc.figure('method_overview.png', 'Fig. 1', diagram['captions']['method_overview'])
    doc.section('Results')
    doc.para('Uneven recall across diagnoses', 'Subheading')
    doc.para(analysis['classification_baseline'].rstrip('.')+' (Table 1; Extended Data Figs. 1–3).')
    doc.table('Table 1', 'Skin lesion classification on 1,512 test images. Precision, recall and F1 are macro averages over seven diagnoses. AUROC is macro one-versus-rest; all other values are percentages.',
              ['Objective','Accuracy','Precision','Recall','F1','AUROC'],
              [[MODEL_NAMES[r['run_id']]]+[percent(r[k]) for k in ('accuracy','macro_precision','macro_recall','macro_f1')]+[f"{float(r['macro_auroc']):.4f}"] for r in cls])
    doc.para(editorial['result_headings']['imbalance'], 'Subheading')
    doc.para(analysis['imbalance'].rstrip('.')+' (Fig. 2).')
    doc.figure('classification_main.png', 'Fig. 2',
               'Class-specific changes under loss rebalancing. a, Recall for seven diagnoses; values in parentheses give test support (n = 1,512 images in total). b, Weighted cross-entropy and focal differences from cross-entropy in percentage points. Symbols show point estimates and lines show paired, true-class-stratified 95% bootstrap intervals from 10,000 resamples. These intervals condition on fitted models and observed class composition and are not multiplicity-adjusted. c,d, Changes in row-normalized confusion relative to cross-entropy, in percentage points, for weighted cross-entropy and focal loss. Rows denote ground truth and columns predicted diagnosis. All three models were trained once from the same initialization.')
    doc.para('High confidence in incorrect predictions', 'Subheading')
    doc.para(editorial['gradcam_summary'].rstrip('.')+' (Fig. 3). [9]')
    doc.figure('classification_gradcam_ce.png', 'Fig. 3',
               'Predicted-class activation in four lesion images. Each case contains the input, activation map and overlay. a,b, Correct NV and MEL predictions. c,d, MEL assigned to NV and VASC assigned to MEL. Images were selected by prediction confidence with distinct reference diagnoses within each correctness group. Heatmaps were min–max normalized independently for each image and cannot compare activation magnitude between images. Displayed probabilities are rounded to three decimal places. NV, naevus; MEL, melanoma; VASC, vascular lesion.')
    doc.para('The dermoscopy examples are adapted from ISIC 2018 Task 3 under the Creative Commons Attribution–NonCommercial 4.0 International licence. The source images were resized and accompanied by model activation maps; dataset attribution and source and licence links are recorded in NOTICE. [15,16]')
    doc.para('Lower mean boundary error with 3D', 'Subheading')
    doc.para(editorial['segmentation_summary'].rstrip('.')+' (Table 2; Fig. 4).')
    doc.table('Table 2', 'Segmentation of 20 complete test volumes. Values are mean ± population standard deviation. Dice and Jaccard are percentages; symmetric average surface distance (ASD) and 95th-percentile Hausdorff distance (HD95) are in voxels. No empty-mask failures occurred.',
              ['Model','Dice','Jaccard','ASD','HD95'],
              [[MODEL_NAMES[r['run_id']]]+[f"{percent(r[k+'_mean'])} ± {percent(r[k+'_std'])}" for k in ('dice','jaccard')]+[f"{float(r[k+'_mean']):.2f} ± {float(r[k+'_std']):.2f}" for k in ('asd','hd95')] for r in seg])
    doc.figure('segmentation_main.png', 'Fig. 4',
               'Paired segmentation outcomes across complete volumes. a–d, Dice, Jaccard, symmetric average surface distance (ASD) and 95th-percentile Hausdorff distance (HD95), respectively. Each pair represents the same test volume (n = 20); lines preserve the pairing. Black horizontal markers indicate means; offset capped intervals show population standard deviations across the 20-volume cohort (ddof = 0), not confidence intervals. The symmetric distance intervals can extend below zero although all individual distances are non-negative. Outlined points identify LA-TE13, the largest 3D HD95. Higher overlap and lower surface distance indicate closer agreement with the reference. Overlap axes cover the observed range; distance units are voxels because physical spacing is unavailable.')
    doc.para('High overlap leaves local errors', 'Subheading')
    doc.para('The poorest 3D result reverses the average model ranking. In LA-TE13, a distant foreground region and an omission give a 3D Dice of 64.12% and HD95 of 65.81 voxels; the 2D Dice is 86.94%. At the other extreme, the highest-Dice 3D volume reaches 94.00% yet omits a small annotated cross-section. The two remaining cases show detached excess foreground and connected overgrowth. In the median-ranked volume, a 1.83% net volume bias conceals both false-positive and false-negative regions.')
    doc.para('Complete-volume measurements and selected-case observations are retained in Supplementary Table 3 and the case records. MRI images and mask illustrations are not distributed in this repository.')
    doc.para('Checkpoint-selection sensitivity', 'Subheading')
    doc.para('An exploratory change to validation macro-F1 selection, with at least 20 training epochs, moved the weighted-loss checkpoint from epoch 2 to 18. Validation macro F1 rose from 74.76% to 80.97%, but test macro F1 increased by only 0.93 percentage points to 67.09%; accuracy decreased to 75.93% and macro recall to 64.50%. Cross-entropy and focal retained their original selected epochs and discrete test metrics. The experiment followed the original test comparison and cannot establish a confirmatory improvement (Extended Data Figs. 5,6; Supplementary Table 5).')
    doc.para('Recall gains reduce class precision', 'Subheading')
    doc.para('The classwise precision–recall positions clarify the cost of increased coverage. For AKIEC, weighted cross-entropy raises recall from 44.19% to 65.12% while precision falls from 73.08% to 43.75%. For DF, recall increases from 63.64% to 79.55%, with precision decreasing from 75.68% to 53.03%. These are fixed argmax operating points rather than threshold-swept curves (Extended Data Fig. 8). Comparison of validation and test outcomes also shows that the selected checkpoint ranking transfers imperfectly between the two partitions (Extended Data Fig. 9); neither the graphical comparison nor the later selection analysis was used to replace the primary models.')
    doc.para('Small mean bias conceals volume errors', 'Subheading')
    doc.para('The mean signed foreground voxel-count bias is +2.91% for 3D and +20.99% for 2D. The median absolute bias is 7.64% and 15.39%, respectively; 13 of 20 3D masks and 5 of 20 2D masks lie within 10% of the reference voxel count. The 2D model overestimates foreground in 19 cases, compared with 9 for 3D. Signed averages can cancel over- and underestimation, and a near-zero bias can coexist with misplaced foreground, as the median-ranked 3D case illustrates. These descriptive comparisons use voxel counts rather than calibrated physical volumes (Extended Data Figs. 10,11).')
    doc.section('Discussion')
    for paragraph in editorial['discussion']:
        doc.para(paragraph)
    doc.para('Choice of a lesion review model', 'Subheading')
    doc.para(analysis['screening'])
    doc.para('Implications for atrial modelling', 'Subheading')
    for paragraph in analysis['anatomical_interpretation']:
        doc.para(paragraph)
    doc.para('Scope of the comparison', 'Subheading')
    doc.para('The segmentation comparison also changes model capacity and training exposure. The 3D model has 5.65 million parameters, versus 1.94 million for 2D, and completed 9,000 versus 12,000 updates at batch sizes of 1 and 16. Shared data and evaluation rules support comparison of the pipelines, but do not isolate neighbouring slices from these differences.')
    doc.para(analysis['limitations'])
    doc.section('Methods', newpage=True)
    expanded = read_json(ROOT/'report/methods_expanded.json')
    for section in expanded:
        doc.para(section['heading'], 'Subheading')
        for block in section['blocks']:
            if block['type'] == 'paragraph':
                doc.para(block['text'])
            elif block['type'] == 'equation':
                doc.equation(block['latex'], block['label'], block.get('description'))
            else:
                raise ValueError('Unknown Methods block')
    doc.para('Additional descriptive comparisons', 'Subheading')
    doc.para('Classwise precision–recall displays show one operating point per fitted classifier and diagnosis, using the fixed multiclass argmax rule. Validation/test alignment joins outcomes for the same selected model across different data partitions; it does not pair the underlying images, provide an uncertainty interval, or change the selection rule. For segmentation, voxel-count bias was derived from each complete prediction and reference mask. The empirical cumulative distribution of absolute bias gives equal weight to each of the 20 volumes. Mean ± SD markers use ddof = 0 throughout, matching the saved summary tables. These displays were added after evaluation as descriptive diagnostics.')
    doc.equation(r'b_i=100\,\frac{|P_i|-|R_i|}{|R_i|},\qquad a_i=|b_i|.', 'eq:voxel-count-bias', 'Here $P_i$ and $R_i$ are the predicted and reference foreground voxel sets for volume $i$. Positive bias denotes overestimation; absolute bias discards the direction. The denominator is nonzero for every supplied test reference mask. These counts cannot establish physical atrial volume without reliable spatial calibration.')
    doc.para('Scientific figure preparation', 'Subheading')
    doc.para('The visualization workflow used the scientific-visualization guidance in Scientific Agent Skills. Figures were drawn from saved predictions and metric tables, retaining raw case values, explicit uncertainty definitions and editable vector text. This guidance concerns figure construction and reporting, rather than model performance. [14]')
    doc.references()
    editorial['captions'].update({
        'unet_architecture.png': diagram['captions']['unet_architecture'],
        'classification_precision_recall_by_class.png': 'Classwise precision and recall at fixed operating points. a–g, The seven diagnoses. Each point represents one model under multiclass argmax prediction; the dashed diagonal marks equal precision and recall. Counts state the test support of each diagnosis. These are single operating points, not precision–recall curves.',
        'classification_validation_test_alignment.png': 'Validation and test outcomes for selected models. a, Accuracy. b, Macro F1. Hollow symbols denote validation (n = 193 images) and filled symbols test (n = 1,512). Lines link the same fitted checkpoint across different partitions and do not represent paired image observations or an uncertainty interval. Original and exploratory checkpoints are shown without further selection.',
        'segmentation_volume_bias.png': 'Foreground voxel-count bias across complete volumes. a, Predicted versus reference counts; the diagonal denotes equality. b, Signed bias versus Dice. c, Absolute bias versus HD95. d, Empirical cumulative distribution of absolute bias. Every test volume is retained (n = 20 per model). Bias is $100(|P|-|R|)/|R|$, with positive values denoting overestimation. Voxel counts are not calibrated physical volumes.',
        'segmentation_metric_summary.png': 'Distribution of complete-volume segmentation outcomes. a–d, Dice, Jaccard, ASD and HD95. All 20 case values are shown per model; black markers give arithmetic means with population standard deviations (ddof = 0). These intervals describe cohort variation and are not confidence intervals. Surface distances use voxels; a symmetric SD interval may extend below zero without any negative observations.',
        'classification_confusion_comparison.png': 'Loss-specific classification errors and shared test support. a–c, Cross-entropy, weighted cross-entropy and focal confusion matrices. Ground-truth diagnoses form rows and predicted diagnoses columns; colours and values give row-normalized percentages on a common scale. d, The test support of each diagnosis (n = 1,512 images in total).',
    })
    doc.section('Extended Data', newpage=True)
    for name,label,height in [
        ('classification_class_distribution.png','Extended Data Fig. 1',85*mm),
        ('classification_learning_curves.png','Extended Data Fig. 2a',80*mm),
        ('classification_roc_ce.png','Extended Data Fig. 2b',95*mm),
        ('classification_confusion_comparison.png','Extended Data Fig. 3',170*mm),
        ('segmentation_learning_curves.png','Extended Data Fig. 4',125*mm),
        ('classification_selection_sensitivity.png','Extended Data Fig. 5',90*mm),
        ('classification_sensitivity_metrics.png','Extended Data Fig. 6',90*mm),
        ('unet_architecture.png','Extended Data Fig. 7',130*mm),
        ('classification_precision_recall_by_class.png','Extended Data Fig. 8',170*mm),
        ('classification_validation_test_alignment.png','Extended Data Fig. 9',130*mm),
        ('segmentation_volume_bias.png','Extended Data Fig. 10',150*mm),
        ('segmentation_metric_summary.png','Extended Data Fig. 11',150*mm),
    ]:
        doc.figure(name,label,editorial['captions'][name],height=height,
                   width=89*mm if name=='classification_roc_ce.png' else 183*mm)
    doc.section('Supplementary tables', newpage=True)
    classes = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
    counts = {(r['split'],r['class_name']):r['count'] for r in read_csv(results/'class_counts.csv')}
    weights = read_json(ROOT/'outputs/cls_r50_wce/class_weights.json')['inverse_weights']
    doc.table('Supplementary Table 1','Class support and inverse-frequency weights. Weights use training counts only.',
              ['Diagnosis','Train','Validation','Test','Weight'],
              [[name,counts['train',name],counts['val',name],counts['test',name],f'{weights[k]:.3f}'] for k,name in enumerate(classes)])
    for i,r in enumerate(cls,1):
        per = read_csv(results/'classification'/r['run_id']/'per_class.csv')
        doc.table(f'Supplementary Table 2{chr(96+i)}',f"Per-class outcomes for {MODEL_NAMES[r['run_id']]}. Precision, recall and F1 are percentages; AUROC uses [0,1].",
                  ['Diagnosis','Support','Precision','Recall','F1','AUROC'],
                  [[x['class_name'],x['support']]+[percent(x[k]) for k in ('precision','recall','f1')]+[f"{float(x['auroc']):.4f}"] for x in per])
    for i,r in enumerate(seg,1):
        doc.new_page()
        cases = read_csv(results/'segmentation'/r['run_id']/'per_case_metrics.csv')
        doc.table(f'Supplementary Table 3{chr(96+i)}',f"Complete-volume outcomes for {MODEL_NAMES[r['run_id']]}. Overlap values are percentages; distances are in voxels.",
                  ['Case','Dice','Jaccard','ASD','HD95'],
                  [[x['case_id']]+[percent(x[k]) for k in ('dice','jaccard')]+[f"{float(x[k]):.2f}" for k in ('asd','hd95')] for x in cases])
    doc.new_page()
    doc.table('Supplementary Table 4','Support-weighted classification outcomes, in percent.',
              ['Objective','Precision','Recall','F1'],
              [[MODEL_NAMES[r['run_id']]]+[percent(r[k]) for k in ('weighted_precision','weighted_recall','weighted_f1')] for r in cls])
    supplementary = read_csv(results/'sensitivity/classification/summary.csv')
    if read_json(results/'sensitivity/validation_checks.json').get('passed') is not True or len(supplementary)!=3:
        raise ValueError('All three verified sensitivity runs are required')
    doc.table('Supplementary Table 5','Exploratory macro-F1 selection. Original selection maximized accuracy. Every epoch was eligible; training completed at least 20 epochs. Test columns are percentages.',
              ['Objective','Selected epoch','Epochs run','Accuracy','Recall','F1'],
              [[MODEL_NAMES[r['run_id'].removesuffix('_sensitivity')],read_json(ROOT/'outputs'/r['run_id']/'metadata.json')['best_epoch'],
                read_json(ROOT/'outputs'/r['run_id']/'metadata.json')['completed_epochs']]+[percent(r[k]) for k in ('accuracy','macro_recall','macro_f1')] for r in supplementary])
    protocols=[]
    for r in cls+seg:
        m=read_json(ROOT/'outputs'/r['run_id']/'metadata.json');c=m['config']
        protocols.append([MODEL_NAMES[r['run_id']],c['batch_size'],str(c.get('patch',[224,224])),
                          m.get('completed_epochs',m.get('completed_updates')),f"{m['parameter_count']/1e6:.2f}",
                          f"{m['training_seconds']/60:.2f}",f"{m['peak_gpu_bytes']/2**30:.2f}"])
    doc.table('Supplementary Table 6','Measured resources for the original experiments. Times include data preparation and shared GPU workload.',
              ['Model','Batch','Input','Epochs/updates','Parameters (M)','Time (min)','Peak (GiB)'],protocols)
    doc.section('Supplementary visual observations')
    for r in analysis['gradcam']+analysis['segmentation_cases']:
        doc.para(r.get('image_id',r.get('case_id')),'Subheading')
        doc.para(r['analysis'])
    output.parent.mkdir(parents=True,exist_ok=True)
    typesetting = doc.build(output, editorial['title'])
    (ROOT/'report/report.md').write_text('\n'.join(doc.markdown))
    save_json(ROOT/'report/data_tables.json',dict(classification=cls,segmentation=seg,supplementary_classification=supplementary,
              protocols=protocols,environment=read_json(results/'environment.json')))
    save_json(ROOT/'report/layout_manifest.json',dict(main_figures=4,extended_data_panels=12,vector_placements=doc.placements,
              citation_order=doc.citation_order,manuscript_style='Research manuscript with expanded Methods and supporting data'))
    print('Built',output)


if __name__=='__main__':
    main()
