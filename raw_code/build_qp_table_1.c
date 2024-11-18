build_qp_table(PPS *pps, int index)

{

    int i;

    for(i = 0; i < 255; i++)

        pps->chroma_qp_table[i & 0xff] = chroma_qp[av_clip(i + index, 0, 51)];

    pps->chroma_qp_index_offset = index;

}
