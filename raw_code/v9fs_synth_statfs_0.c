static int v9fs_synth_statfs(FsContext *s, V9fsPath *fs_path,

                             struct statfs *stbuf)

{

    stbuf->f_type = 0xABCD;

    stbuf->f_bsize = 512;

    stbuf->f_blocks = 0;

    stbuf->f_files = v9fs_synth_node_count;

    stbuf->f_namelen = NAME_MAX;

    return 0;

}
