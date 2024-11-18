static int v9fs_synth_opendir(FsContext *ctx,

                             V9fsPath *fs_path, V9fsFidOpenState *fs)

{

    V9fsSynthOpenState *synth_open;

    V9fsSynthNode *node = *(V9fsSynthNode **)fs_path->data;



    synth_open = g_malloc(sizeof(*synth_open));

    synth_open->node = node;

    node->open_count++;

    fs->private = synth_open;

    return 0;

}
