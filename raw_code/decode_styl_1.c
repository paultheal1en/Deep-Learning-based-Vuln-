static int decode_styl(const uint8_t *tsmb, MovTextContext *m, AVPacket *avpkt)

{

    int i;

    m->style_entries = AV_RB16(tsmb);

    tsmb += 2;

    // A single style record is of length 12 bytes.

    if (m->tracksize + m->size_var + 2 + m->style_entries * 12 > avpkt->size)

        return -1;



    m->box_flags |= STYL_BOX;

    for(i = 0; i < m->style_entries; i++) {

        m->s_temp = av_malloc(sizeof(*m->s_temp));

        if (!m->s_temp) {

            mov_text_cleanup(m);

            return AVERROR(ENOMEM);

        }

        m->s_temp->style_start = AV_RB16(tsmb);

        tsmb += 2;

        m->s_temp->style_end = AV_RB16(tsmb);

        tsmb += 2;

        m->s_temp->style_fontID = AV_RB16(tsmb);

        tsmb += 2;

        m->s_temp->style_flag = AV_RB8(tsmb);

        tsmb++;

        m->s_temp->fontsize = AV_RB8(tsmb);

        av_dynarray_add(&m->s, &m->count_s, m->s_temp);

        if(!m->s) {

            mov_text_cleanup(m);

            return AVERROR(ENOMEM);

        }

        tsmb++;

        // text-color-rgba

        tsmb += 4;

    }

    return 0;

}
