import os
import sys
now_dir = os.getcwd()
sys.path.append(now_dir)
import torch
from fairseq import checkpoint_utils
import scipy.io.wavfile as wavfile
import faiss
from slicer2 import Slicer
from scipy import signal
import numpy as np
import traceback
#import wandb
from config import DeviceConfig
from lib.infer_pack.models import SynthesizerTrnMs768NSFsid
from my_utils import load_audio
from vc_infer_pipeline import VC

#wandb.init(project="mangio-infer-batch", sync_tensorboard=True)

DoFormant, Quefrency, Timbre = False, 1.0, 1.0
config = DeviceConfig()

def load_hubert():
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.float()
    hubert_model.eval()
    return hubert_model

def files(path):
    if os.path.isdir(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield (path, file)
    if os.path.isfile(path):
        yield (os.path.dirname(path), os.path.basename(path))

# Pour chaque modèle,
#   Pour chaque transposition/formant
#      Pour chaque fichier
#        Pour chaque params


def cli():
    import argparse
    parser = argparse.ArgumentParser(
                    prog='rvc-cli',
                    description='Mangio RVC CLI',
                    epilog='This is like infer-web.py --is_cli but not insane')
    parser.add_argument('--model', action='append', required=True) # Models
    parser.add_argument('--index', required=True)
    parser.add_argument('--transpose', action='append', required=True) # Transposes
    parser.add_argument('--vol', action='append', required=True) # Volume enveloppe
    parser.add_argument('--method', action='append', required=True) # Methods
    parser.add_argument('--protect', action='append', required=True)
    parser.add_argument('--feature', action='append', required=True)
    parser.add_argument('--harvest_radius', action='append')
    parser.add_argument('--crepe_hop', action='append')
    parser.add_argument('input')
    args = parser.parse_args()
    print(args)
    hubert = load_hubert()
    try:
        index = faiss.read_index(args.index)
        # big_npy = np.load(file_big_npy)
        big_npy = index.reconstruct_n(0, index.ntotal)
    except:
        traceback.print_exc()
        index = big_npy = None
    for m in args.model:
        print(f"[LOAD ] model={m}")
        # like infer-web.py:get_wc but not retarded
        feature_index_path = args.index
        cpt = torch.load(m, map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        if if_f0 == 0:
            raise Exception("This doesn't work for non-pitch models")
        version = cpt.get("version", "v1")
        if version == "v1":
            raise Exception("This doesn't work with v1 models")
        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        del net_g.enc_q
        print(net_g.load_state_dict(cpt["weight"], strict=False))
        net_g.eval().to(config.device)
        if config.is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()
        vc = VC(tgt_sr, config)
        n_spk = cpt["config"][-3]
        global_output_dir = "outputs"
        model_name = os.path.basename(m)
        os.makedirs(global_output_dir, exist_ok=True)
        for (path, input) in files(args.input):
            print(f"[INPUT] model={m} path={path} input={input}")
            output_dir = "%s/%s" % (global_output_dir, os.path.basename(os.path.dirname(path)))
            os.makedirs(output_dir, exist_ok=True)
            print(f"root out dir: {output_dir}")
            output_dir = "%s/%s" % (output_dir, os.path.splitext(os.path.basename(input))[0])
            os.makedirs(output_dir, exist_ok=True)
            print(f"file out dir: {output_dir}")

            full_audio = load_audio(os.path.join(path, input), 16000, DoFormant, Quefrency, Timbre)
            bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
            audio = signal.lfilter(bh, ah, full_audio)
            #wavfile.write("%s/input.wav" % output_dir, 16000, audio)
            slicer = Slicer(
            sr=16000,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
            )
            idx = 0
            for audio in slicer.slice(audio):
            #for audio in [full_audio]:
                idx += 1
                audio_max = np.abs(audio).max() / 0.95
                if audio_max > 1:
                    audio /= audio_max

                wavfile.write("%s/input p%s.wav" % (output_dir, idx), 16000, audio)

                times = [0, 0, 0]
                if_f0 = cpt.get("f0", 1)
                (audio_pad, opt_ts, inp_f0, p_len, t1) = vc.audio_pad(audio)
                for vol in args.vol:
                    for method in args.method:

                        # if harvest, radiis
                        if "harvest" in method:
                            h_m_fs = args.harvest_radius
                        else:
                            h_m_fs = [0]

                        for harvest_median_filter in h_m_fs:
                            # if crepe, hops
                            if "crepe" in method:
                                hops = args.crepe_hop
                            else:
                                hops = [0]

                            for crepe_hop_length in hops:

                                for t in args.transpose:

                                    # Build f0 for input here
                                    print(f"[PITCH] model={m} input={input} part={idx} transpose={t} vol={vol} method={method} harvest-median-filer={harvest_median_filter} crepe-hop={crepe_hop_length}")

                                    (pitch, pitchf, t2) = vc.do_pitch(if_f0, input, audio_pad, p_len,
                                                                    int(t), method, int(harvest_median_filter), int(crepe_hop_length), inp_f0)

                                    for feature in args.feature:
                                        for protect in args.protect:
                                            print(f"[VC   ] model={m} input={input} part={idx}  transpose={t} vol={vol} method={method} feature={feature} protect={protect} harvest-median-filer={harvest_median_filter} crepe-hop={crepe_hop_length}")

                                            output_file = f"{model_name} - p{idx} - {t} - {method} [vol={vol},feat={feature},protect={protect},mf={harvest_median_filter},hop={crepe_hop_length}].wav"
                                            output_file = "%s/%s" % (output_dir, output_file)

                                            if os.path.isfile(output_file):
                                                print(f"- skipping, already existing: {output_file}")
                                                (True, output_file, None, None)
                                                continue
                        
                                            speaker_id = 0
                                            transposition = int(t)
                                            f0_method = method
                                            crepe_hop_length = int(crepe_hop_length)
                                            harvest_median_filter = int(harvest_median_filter)
                                            resample = 0
                                            mix = float(vol)
                                            feature_ratio = float(feature)
                                            protection_amnt = float(protect)
                                            protect1 = 0.5

                                            f0_file = None

                                            try:
                                                audio_opt = vc.pipeline(
                                                    hubert,
                                                    net_g,
                                                    speaker_id,
                                                    audio,
                                                    os.path.join(path, input), #input_audio_path1,
                                                    times,
                                                    transposition,
                                                    f0_method,
                                                    index,
                                                    big_npy,
                                                    feature_ratio,
                                                    if_f0,
                                                    harvest_median_filter,
                                                    tgt_sr,
                                                    resample,
                                                    mix,
                                                    version,
                                                    protection_amnt,
                                                    crepe_hop_length,
                                                    f0_file=f0_file,
                                                    #audio_pad=None,
                                                    #pitch=None
                                                    audio_pad=(audio_pad, opt_ts, inp_f0, p_len, t1),
                                                    pitch=(pitch, pitchf, t2)
                                                )
                                                wavfile.write(
                                                    output_file,
                                                    tgt_sr,
                                                    audio_opt,
                                                )
                                                print(f"- wrote {output_file}")
                                                (True, output_file, tgt_sr, audio_opt)
                                                continue
                                            except:
                                                info = traceback.format_exc()
                                                print(info)
                                                (False, None, None, None)
                                                continue
                                    
                                    del pitch, pitchf
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()

    print("all done")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(cli())