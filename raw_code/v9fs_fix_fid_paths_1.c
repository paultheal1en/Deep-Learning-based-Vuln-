static void coroutine_fn v9fs_fix_fid_paths(V9fsPDU *pdu, V9fsPath *olddir,

                                            V9fsString *old_name,

                                            V9fsPath *newdir,

                                            V9fsString *new_name)

{

    V9fsFidState *tfidp;

    V9fsPath oldpath, newpath;

    V9fsState *s = pdu->s;





    v9fs_path_init(&oldpath);

    v9fs_path_init(&newpath);

    v9fs_co_name_to_path(pdu, olddir, old_name->data, &oldpath);

    v9fs_co_name_to_path(pdu, newdir, new_name->data, &newpath);



    /*

     * Fixup fid's pointing to the old name to

     * start pointing to the new name

     */

    for (tfidp = s->fid_list; tfidp; tfidp = tfidp->next) {

        if (v9fs_path_is_ancestor(&oldpath, &tfidp->path)) {

            /* replace the name */

            v9fs_fix_path(&tfidp->path, &newpath, strlen(oldpath.data));

        }

    }

    v9fs_path_free(&oldpath);

    v9fs_path_free(&newpath);

}
