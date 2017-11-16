import consts as ct
from audioclip import AudioClip

def ac_setup2():
    e1 = AudioClip(ct.TRAINDIR + "vowels/e2/e2_1.wav",
            mintime=5)
    e1.rsetup([0, '0', 0.6, 'e2', 0.95, '0'])

    e2 = AudioClip(ct.TRAINDIR + "vowels/e2/e2_2.wav",
            mintime=5)
    e2.rsetup([0, '0', 0.6, 'e2', 0.92, '0'])

    e3 = AudioClip(ct.TRAINDIR + "vowels/e2/e2_3.wav",
            mintime=5)
    e3.rsetup([0, '0', 0.49, 'e2', 0.82, '0'])

    e4 = AudioClip(ct.TRAINDIR + "vowels/e2/e2_4.wav",
            mintime=5)
    e4.rsetup([0, '0', 2.65, 'e2', 2.98, '0'])

    ai1 = AudioClip(ct.TRAINDIR + "vowels/ai/ai_1.wav",
            mintime=5)
    ai1.rsetup([0, '0', 0.57, 'ai', 0.94, '0'])

    ai2 = AudioClip(ct.TRAINDIR + "vowels/ai/ai_2.wav",
            mintime=5)
    ai2.rsetup([0, '0', 0.445, 'ai', 0.820, '0'])

    ai3 = AudioClip(ct.TRAINDIR + "vowels/ai/ai_3.wav",
            mintime=5)
    ai3.rsetup([0, '0', 1.03, 'ai', 1.49, '0'])

    #ai4 = AudioClip("vowels/ai/ai_4.wav",
    #        mintime=5)
    #ai4.rsetup([0, '0', 1.91, 'ai', 2.33, '0'])

    return [e1, e2, e3, e4, ai1, ai2, ai3]
    #return [e1, e2, e3, e4, ai1, ai2, ai3, ai4]

def ac_setup3():
    sil = AudioClip(ct.TRAINDIR + "vowels/e2/e2_1.wav",
            start=0,
            end=0.6,
            mintime=5)
    e1 = AudioClip(ct.TRAINDIR + "vowels/e2/e2_1.wav",
            start=0.7,
            end=0.95,
            mintime=5)
    e1.rsetup([0, '0', 0.6, 'e2', 0.95, '0'])

    e2 = AudioClip(ct.TRAINDIR + "vowels/e2/e2_2.wav",
            mintime=5)
    e2.rsetup([0, '0', 0.6, 'e2', 0.92, '0'])

    e3 = AudioClip(ct.TRAINDIR + "vowels/e2/e2_3.wav",
            mintime=5)
    e3.rsetup([0, '0', 0.49, 'e2', 0.82, '0'])

    e4 = AudioClip(ct.TRAINDIR + "vowels/e2/e2_4.wav",
            mintime=5)
    e4.rsetup([0, '0', 2.65, 'e2', 2.98, '0'])

    ai1 = AudioClip(ct.TRAINDIR + "vowels/ai/ai_1.wav",
            mintime=5)
    ai1.rsetup([0, '0', 0.57, 'ai', 0.94, '0'])

    ai2 = AudioClip(ct.TRAINDIR + "vowels/ai/ai_2.wav",
            mintime=5)
    ai2.rsetup([0, '0', 0.445, 'ai', 0.820, '0'])

    ai3 = AudioClip(ct.TRAINDIR + "vowels/ai/ai_3.wav",
            mintime=5)
    ai3.rsetup([0, '0', 1.03, 'ai', 1.49, '0'])

    #ai4 = AudioClip("vowels/ai/ai_4.wav",
    #        mintime=5)
    #ai4.rsetup([0, '0', 1.91, 'ai', 2.33, '0'])

    return [e1, e2, e3, e4, ai1, ai2, ai3]
    #return [e1, e2, e3, e4, ai1, ai2, ai3, ai4]
def ac_setup1():
    ac1 = AudioClip("some_vowels.wav",
            start=0, end=6, mintime=7)
    ac1.region_setup(
            [0, 0.581, 1.103, 2.0, 2.61, 3.51, 4.03, 5.039, 5.55],
            ['0', 'ai', '0', 'ai', '0', 'ai', '0', 'ai', '0'])

    ac2 = AudioClip("more_vowels.wav",
            start=0, end=7, mintime=7)
    ac2.rsetup([
        0, '0', 0.43, 'ai', 0.8, '0', 2.07, 'ai', 2.57, '0', 5.49, 'ai', 5.96,
        '0'])

    ac3 = AudioClip("e.wav",
            start=0, end=6.5, mintime=7)
    ac3.rsetup([
        0, '0', 0.52, 'e2', 1.04, '0', 2.18, 'e2', 2.75, '0', 3.97, 'e2',
        4.53, '0', 5.63, 'e2', 6.27, '0'])

    return [ac1, ac2, ac3]
