static void celt_search_for_dual_stereo(OpusPsyContext *s, CeltFrame *f)
{
    float td1, td2;
    f->dual_stereo = 0;
    bands_dist(s, f, &td1);
    f->dual_stereo = 1;
    bands_dist(s, f, &td2);
    f->dual_stereo = td2 < td1;
    s->dual_stereo_used += td2 < td1;
}