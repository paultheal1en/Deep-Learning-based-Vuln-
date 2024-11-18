static void vnc_async_encoding_end(VncState *orig, VncState *local)
{
    orig->tight = local->tight;
    orig->zlib = local->zlib;
    orig->hextile = local->hextile;
    orig->zrle = local->zrle;
    orig->lossy_rect = local->lossy_rect;
}