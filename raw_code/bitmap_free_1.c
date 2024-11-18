static void bitmap_free(Qcow2Bitmap *bm)
{
    g_free(bm->name);
    g_free(bm);