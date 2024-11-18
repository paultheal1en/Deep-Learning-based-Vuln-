static int v9fs_synth_close(FsContext *ctx, V9fsFidOpenState *fs)

{

    V9fsSynthOpenState *synth_open = fs->private;

    V9fsSynthNode *node = synth_open->node;



    node->open_count--;

    g_free(synth_open);

    fs->private = NULL;

    return 0;

}
