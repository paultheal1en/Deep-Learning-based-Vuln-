static void v9fs_synth_fill_statbuf(V9fsSynthNode *node, struct stat *stbuf)

{

    stbuf->st_dev = 0;

    stbuf->st_ino = node->attr->inode;

    stbuf->st_mode = node->attr->mode;

    stbuf->st_nlink = node->attr->nlink;

    stbuf->st_uid = 0;

    stbuf->st_gid = 0;

    stbuf->st_rdev = 0;

    stbuf->st_size = 0;

    stbuf->st_blksize = 0;

    stbuf->st_blocks = 0;

    stbuf->st_atime = 0;

    stbuf->st_mtime = 0;

    stbuf->st_ctime = 0;

}
