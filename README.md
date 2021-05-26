

# StarGAN Voice Conversion

### Repo: Project for DD2424

### Paper and Dataset

**Paper：**[STARGAN-VC: NON-PARALLEL MANY-TO-MANY VOICE CONVERSION WITH STAR GENERATIVE ADVERSARIAL NETWORKS    ](https://ieeexplore.ieee.org/abstract/document/8639535?casa_token=P5WcObrzJPYAAAAA:vg_YXXJOpbx-Aw5vLb2LyHQlR6GMdEisNrIPVN_MGZVDLfna_NAxd3KNC2ONSlFdEAwE79Q)

**Dataset：**[VCC2018](https://erepo.uef.fi/handle/123456789/7185)

<br/>

### Model Structure

![image](https://github.com/yuexin001/StarGAN-Voice-Conversion/raw/master/StarGAN.png)

<br/>

### File Structure

```bash
|--convert.py
|--model.py
|--module.py
|--preprocess.py
|--train.py
|--utility.py
|--data--|fourspeakers
       --|fourspeakers_test
```

Speakers SF1, SF2, TM1, TM2 are used in this experiment.

<br/>

### Usage

#### Preprocess

```python
python preprocess.py
```

<br/>

#### Train

```python
python train.py
```

<br/>

#### Inference

```python
python convert.py
```





