static void v9fs_wstat_post_chown(V9fsState *s, V9fsWstatState *vs, int err)

{

    V9fsFidState *fidp;

    if (err < 0) {

        goto out;

    }



    if (vs->v9stat.name.size != 0) {

        char *old_name, *new_name;

        char *end;



        old_name = vs->fidp->path.data;

        end = strrchr(old_name, '/');

        if (end) {

            end++;

        } else {

            end = old_name;

        }



        new_name = qemu_mallocz(end - old_name + vs->v9stat.name.size + 1);



        memcpy(new_name, old_name, end - old_name);

        memcpy(new_name + (end - old_name), vs->v9stat.name.data,

                vs->v9stat.name.size);

        vs->nname.data = new_name;

        vs->nname.size = strlen(new_name);



        if (strcmp(new_name, vs->fidp->path.data) != 0) {

            if (v9fs_do_rename(s, &vs->fidp->path, &vs->nname)) {

                err = -errno;

            } else {

                /*

                 * Fixup fid's pointing to the old name to

                 * start pointing to the new name

                 */

                for (fidp = s->fid_list; fidp; fidp = fidp->next) {



                    if (vs->fidp == fidp) {

                        /*

                         * we replace name of this fid towards the end

                         * so that our below strcmp will work

                         */

                        continue;

                    }

                    if (!strncmp(vs->fidp->path.data, fidp->path.data,

                                 strlen(vs->fidp->path.data))) {

                        /* replace the name */

                        v9fs_fix_path(&fidp->path, &vs->nname,

                                      strlen(vs->fidp->path.data));

                    }

                }

                v9fs_string_copy(&vs->fidp->path, &vs->nname);

            }

        }

    }

    v9fs_wstat_post_rename(s, vs, err);

    return;



out:

    v9fs_stat_free(&vs->v9stat);

    complete_pdu(s, vs->pdu, err);

    qemu_free(vs);

}
