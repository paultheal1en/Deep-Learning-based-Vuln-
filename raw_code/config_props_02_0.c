static int config_props(AVFilterLink *link)

{

    UnsharpContext *unsharp = link->dst->priv;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(link->format);



    unsharp->hsub = desc->log2_chroma_w;

    unsharp->vsub = desc->log2_chroma_h;



    init_filter_param(link->dst, &unsharp->luma,   "luma",   link->w);

    init_filter_param(link->dst, &unsharp->chroma, "chroma", SHIFTUP(link->w, unsharp->hsub));



    return 0;

}
