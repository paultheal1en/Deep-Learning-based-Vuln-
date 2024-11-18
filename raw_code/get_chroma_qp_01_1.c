static inline int get_chroma_qp(H264Context *h, int qscale){

    return h->pps.chroma_qp_table[qscale & 0xff];

}
