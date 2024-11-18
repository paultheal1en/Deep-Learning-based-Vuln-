static void v9fs_synth_direntry(V9fsSynthNode *node,

                                struct dirent *entry, off_t off)

{

    strcpy(entry->d_name, node->name);

    entry->d_ino = node->attr->inode;

    entry->d_off = off + 1;

}
