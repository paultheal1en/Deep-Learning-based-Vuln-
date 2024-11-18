static void vmdaudio_loadsound(VmdAudioContext *s, unsigned char *data,

    uint8_t *buf, int silence)

{

    if (s->channels == 2) {

        if ((s->block_align & 0x01) == 0) {

            if (silence)

                memset(data, 0, s->block_align * 2);

            else

                vmdaudio_decode_audio(s, data, buf, 1);

        } else {

            if (silence)

                memset(data, 0, s->block_align * 2);

//            else

//                vmdaudio_decode_audio(s, data, buf, 1);

        }

    } else {

    }

}
