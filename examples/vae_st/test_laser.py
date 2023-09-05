from fairseq.models.wav2vec.wav2vec2_laser import Wav2VecLaser
import torch.nn.functional as F
import torch
import soundfile as sf

model = Wav2VecLaser.from_pretrained('/media/speech/304d52c6-17e2-43b3-a344-fa3686f7638c/MuST-C', checkpoint_file='english.pt').models[0]
model.cuda()
model.eval()
wav, sr = sf.read('14690_0000000-0017000.wav') # sr needs to be 16000
feats = torch.from_numpy(wav).float()


with torch.no_grad():
    feats = F.layer_norm(feats, feats.shape).unsqueeze(0).cuda()
    padding_mask = torch.Tensor([False]*feats.shape[1]).cuda()
    sample = {'padding_mask': padding_mask, 'source': feats}
    embedding = model(**sample)
    print(embedding.size())
    sd
