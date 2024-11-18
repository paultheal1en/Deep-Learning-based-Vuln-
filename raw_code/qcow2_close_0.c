static void qcow2_close(BlockDriverState *bs)

{

    BDRVQcow2State *s = bs->opaque;

    qemu_vfree(s->l1_table);

    /* else pre-write overlap checks in cache_destroy may crash */

    s->l1_table = NULL;



    if (!(s->flags & BDRV_O_INACTIVE)) {

        qcow2_inactivate(bs);

    }



    cache_clean_timer_del(bs);

    qcow2_cache_destroy(bs, s->l2_table_cache);

    qcow2_cache_destroy(bs, s->refcount_block_cache);



    qcrypto_cipher_free(s->cipher);

    s->cipher = NULL;



    g_free(s->unknown_header_fields);

    cleanup_unknown_header_ext(bs);



    g_free(s->image_backing_file);

    g_free(s->image_backing_format);



    g_free(s->cluster_cache);

    qemu_vfree(s->cluster_data);

    qcow2_refcount_close(bs);

    qcow2_free_snapshots(bs);

}
