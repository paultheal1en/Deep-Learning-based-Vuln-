static void *ff_avio_child_next(void *obj, void *prev)

{

    AVIOContext *s = obj;

    AVIOInternal *internal = s->opaque;

    return prev ? NULL : internal->h;

}
