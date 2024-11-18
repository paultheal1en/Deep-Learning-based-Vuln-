static void bdrv_inherited_options(int *child_flags, QDict *child_options,

                                   int parent_flags, QDict *parent_options)

{

    int flags = parent_flags;



    /* Enable protocol handling, disable format probing for bs->file */

    flags |= BDRV_O_PROTOCOL;



    /* If the cache mode isn't explicitly set, inherit direct and no-flush from

     * the parent. */

    qdict_copy_default(child_options, parent_options, BDRV_OPT_CACHE_DIRECT);

    qdict_copy_default(child_options, parent_options, BDRV_OPT_CACHE_NO_FLUSH);



    /* Inherit the read-only option from the parent if it's not set */

    qdict_copy_default(child_options, parent_options, BDRV_OPT_READ_ONLY);



    /* Our block drivers take care to send flushes and respect unmap policy,

     * so we can default to enable both on lower layers regardless of the

     * corresponding parent options. */

    flags |= BDRV_O_UNMAP;



    /* Clear flags that only apply to the top layer */

    flags &= ~(BDRV_O_SNAPSHOT | BDRV_O_NO_BACKING | BDRV_O_COPY_ON_READ |

               BDRV_O_NO_IO);



    *child_flags = flags;

}
