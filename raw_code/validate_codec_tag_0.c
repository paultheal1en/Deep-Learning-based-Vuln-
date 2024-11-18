static int validate_codec_tag(AVFormatContext *s, AVStream *st)

{

    const AVCodecTag *avctag;

    int n;

    enum AVCodecID id = AV_CODEC_ID_NONE;

    unsigned int tag  = 0;



    /**

     * Check that tag + id is in the table

     * If neither is in the table -> OK

     * If tag is in the table with another id -> FAIL

     * If id is in the table with another tag -> FAIL unless strict < normal

     */

    for (n = 0; s->oformat->codec_tag[n]; n++) {

        avctag = s->oformat->codec_tag[n];

        while (avctag->id != AV_CODEC_ID_NONE) {

            if (avpriv_toupper4(avctag->tag) == avpriv_toupper4(st->codec->codec_tag)) {

                id = avctag->id;

                if (id == st->codec->codec_id)

                    return 1;

            }

            if (avctag->id == st->codec->codec_id)

                tag = avctag->tag;

            avctag++;

        }

    }

    if (id != AV_CODEC_ID_NONE)

        return 0;

    if (tag && (st->codec->strict_std_compliance >= FF_COMPLIANCE_NORMAL))

        return 0;

    return 1;

}
