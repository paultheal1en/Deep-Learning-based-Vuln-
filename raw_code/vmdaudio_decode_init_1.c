static int vmdaudio_decode_init(AVCodecContext *avctx)

{

    VmdAudioContext *s = (VmdAudioContext *)avctx->priv_data;

    int i;



    s->channels = avctx->channels;

    s->bits = avctx->bits_per_sample;

    s->block_align = avctx->block_align;



printf ("  %d channels, %d bits/sample, block align = %d\n",

  s->channels, s->bits, s->block_align);



    /* set up the steps8 and steps16 tables */

    for (i = 0; i < 8; i++) {

        if (i < 4)

            s->steps8[i] = i;

        else

            s->steps8[i] = s->steps8[i - 1] + i - 1;



        if (i == 0)

            s->steps16[i] = 0;

        else if (i == 1)

            s->steps16[i] = 4;

        else if (i == 2)

            s->steps16[i] = 16;

        else

            s->steps16[i] = 1 << (i + 4);

    }



    /* set up the step128 table */

    s->steps128[0] = 0;

    s->steps128[1] = 8;

    for (i = 0x02; i <= 0x20; i++)

        s->steps128[i] = (i - 1) << 4;

    for (i = 0x21; i <= 0x60; i++)

        s->steps128[i] = (i + 0x1F) << 3;

    for (i = 0x61; i <= 0x70; i++)

        s->steps128[i] = (i - 0x51) << 6;

    for (i = 0x71; i <= 0x78; i++)

        s->steps128[i] = (i - 0x69) << 8;

    for (i = 0x79; i <= 0x7D; i++)

        s->steps128[i] = (i - 0x75) << 10;

    s->steps128[0x7E] = 0x3000;

    s->steps128[0x7F] = 0x4000;



    /* set up the negative half of each table */

    for (i = 0; i < 8; i++) {

        s->steps8[i + 8] = -s->steps8[i];

        s->steps16[i + 8] = -s->steps16[i];

    }

    for (i = 0; i < 128; i++)

      s->steps128[i + 128] = -s->steps128[i];



    return 0;

}
