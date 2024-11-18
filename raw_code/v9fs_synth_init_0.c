static int v9fs_synth_init(FsContext *ctx)

{

    QLIST_INIT(&v9fs_synth_root.child);

    qemu_mutex_init(&v9fs_synth_mutex);



    /* Add "." and ".." entries for root */

    v9fs_add_dir_node(&v9fs_synth_root, v9fs_synth_root.attr->mode,

                      "..", v9fs_synth_root.attr, v9fs_synth_root.attr->inode);

    v9fs_add_dir_node(&v9fs_synth_root, v9fs_synth_root.attr->mode,

                      ".", v9fs_synth_root.attr, v9fs_synth_root.attr->inode);



    /* Mark the subsystem is ready for use */

    v9fs_synth_fs = 1;

    return 0;

}
