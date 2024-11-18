static uint32_t get_cluster_count_for_direntry(BDRVVVFATState* s,

	direntry_t* direntry, const char* path)

{

    /*

     * This is a little bit tricky:

     * IF the guest OS just inserts a cluster into the file chain,

     * and leaves the rest alone, (i.e. the original file had clusters

     * 15 -> 16, but now has 15 -> 32 -> 16), then the following happens:

     *

     * - do_commit will write the cluster into the file at the given

     *   offset, but

     *

     * - the cluster which is overwritten should be moved to a later

     *   position in the file.

     *

     * I am not aware that any OS does something as braindead, but this

     * situation could happen anyway when not committing for a long time.

     * Just to be sure that this does not bite us, detect it, and copy the

     * contents of the clusters to-be-overwritten into the qcow.

     */

    int copy_it = 0;

    int was_modified = 0;

    int32_t ret = 0;



    uint32_t cluster_num = begin_of_direntry(direntry);

    uint32_t offset = 0;

    int first_mapping_index = -1;

    mapping_t* mapping = NULL;

    const char* basename2 = NULL;



    vvfat_close_current_file(s);



    /* the root directory */

    if (cluster_num == 0)

	return 0;



    /* write support */

    if (s->qcow) {

	basename2 = get_basename(path);



	mapping = find_mapping_for_cluster(s, cluster_num);



	if (mapping) {

	    const char* basename;



	    assert(mapping->mode & MODE_DELETED);

	    mapping->mode &= ~MODE_DELETED;



	    basename = get_basename(mapping->path);



	    assert(mapping->mode & MODE_NORMAL);



	    /* rename */

	    if (strcmp(basename, basename2))

		schedule_rename(s, cluster_num, strdup(path));

	} else if (is_file(direntry))

	    /* new file */

	    schedule_new_file(s, strdup(path), cluster_num);

	else {

	    assert(0);

	    return 0;

	}

    }



    while(1) {

	if (s->qcow) {

	    if (!copy_it && cluster_was_modified(s, cluster_num)) {

		if (mapping == NULL ||

			mapping->begin > cluster_num ||

			mapping->end <= cluster_num)

		mapping = find_mapping_for_cluster(s, cluster_num);





		if (mapping &&

			(mapping->mode & MODE_DIRECTORY) == 0) {



		    /* was modified in qcow */

		    if (offset != mapping->info.file.offset + s->cluster_size

			    * (cluster_num - mapping->begin)) {

			/* offset of this cluster in file chain has changed */

			assert(0);

			copy_it = 1;

		    } else if (offset == 0) {

			const char* basename = get_basename(mapping->path);



			if (strcmp(basename, basename2))

			    copy_it = 1;

			first_mapping_index = array_index(&(s->mapping), mapping);

		    }



		    if (mapping->first_mapping_index != first_mapping_index

			    && mapping->info.file.offset > 0) {

			assert(0);

			copy_it = 1;

		    }



		    /* need to write out? */

		    if (!was_modified && is_file(direntry)) {

			was_modified = 1;

			schedule_writeout(s, mapping->dir_index, offset);

		    }

		}

	    }



	    if (copy_it) {

		int i, dummy;

		/*

		 * This is horribly inefficient, but that is okay, since

		 * it is rarely executed, if at all.

		 */

		int64_t offset = cluster2sector(s, cluster_num);



		vvfat_close_current_file(s);

		for (i = 0; i < s->sectors_per_cluster; i++)

		    if (!s->qcow->drv->bdrv_is_allocated(s->qcow,

				offset + i, 1, &dummy)) {

			if (vvfat_read(s->bs,

				    offset, s->cluster_buffer, 1))

			    return -1;

			if (s->qcow->drv->bdrv_write(s->qcow,

				    offset, s->cluster_buffer, 1))

			    return -2;

		    }

	    }

	}



	ret++;

	if (s->used_clusters[cluster_num] & USED_ANY)

	    return 0;

	s->used_clusters[cluster_num] = USED_FILE;



	cluster_num = modified_fat_get(s, cluster_num);



	if (fat_eof(s, cluster_num))

	    return ret;

	else if (cluster_num < 2 || cluster_num > s->max_fat_value - 16)

	    return -1;



	offset += s->cluster_size;

    }

}
