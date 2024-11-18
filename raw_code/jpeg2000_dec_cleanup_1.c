static void jpeg2000_dec_cleanup(Jpeg2000DecoderContext *s)

{

    int tileno, compno;

    for (tileno = 0; tileno < s->numXtiles * s->numYtiles; tileno++) {

        if (s->tile[tileno].comp) {

            for (compno = 0; compno < s->ncomponents; compno++) {

                Jpeg2000Component *comp     = s->tile[tileno].comp   + compno;

                Jpeg2000CodingStyle *codsty = s->tile[tileno].codsty + compno;



                ff_jpeg2000_cleanup(comp, codsty);

            }

            av_freep(&s->tile[tileno].comp);

        }

    }

    av_freep(&s->tile);



    s->numXtiles = s->numYtiles = 0;

}