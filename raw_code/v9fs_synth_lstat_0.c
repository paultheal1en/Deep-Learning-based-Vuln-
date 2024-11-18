static int v9fs_synth_lstat(FsContext *fs_ctx,

                            V9fsPath *fs_path, struct stat *stbuf)

{

    V9fsSynthNode *node = *(V9fsSynthNode **)fs_path->data;



    v9fs_synth_fill_statbuf(node, stbuf);

    return 0;

}
