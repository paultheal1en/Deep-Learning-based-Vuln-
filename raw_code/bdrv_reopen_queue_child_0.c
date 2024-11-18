static BlockReopenQueue *bdrv_reopen_queue_child(BlockReopenQueue *bs_queue,

                                                 BlockDriverState *bs,

                                                 QDict *options,

                                                 int flags,

                                                 const BdrvChildRole *role,

                                                 QDict *parent_options,

                                                 int parent_flags)

{

    assert(bs != NULL);



    BlockReopenQueueEntry *bs_entry;

    BdrvChild *child;

    QDict *old_options, *explicit_options;



    if (bs_queue == NULL) {

        bs_queue = g_new0(BlockReopenQueue, 1);

        QSIMPLEQ_INIT(bs_queue);

    }



    if (!options) {

        options = qdict_new();

    }



    /*

     * Precedence of options:

     * 1. Explicitly passed in options (highest)

     * 2. Set in flags (only for top level)

     * 3. Retained from explicitly set options of bs

     * 4. Inherited from parent node

     * 5. Retained from effective options of bs

     */



    if (!parent_options) {

        /*

         * Any setting represented by flags is always updated. If the

         * corresponding QDict option is set, it takes precedence. Otherwise

         * the flag is translated into a QDict option. The old setting of bs is

         * not considered.

         */

        update_options_from_flags(options, flags);

    }



    /* Old explicitly set values (don't overwrite by inherited value) */

    old_options = qdict_clone_shallow(bs->explicit_options);

    bdrv_join_options(bs, options, old_options);

    QDECREF(old_options);



    explicit_options = qdict_clone_shallow(options);



    /* Inherit from parent node */

    if (parent_options) {

        assert(!flags);

        role->inherit_options(&flags, options, parent_flags, parent_options);

    }



    /* Old values are used for options that aren't set yet */

    old_options = qdict_clone_shallow(bs->options);

    bdrv_join_options(bs, options, old_options);

    QDECREF(old_options);



    /* bdrv_open() masks this flag out */

    flags &= ~BDRV_O_PROTOCOL;



    QLIST_FOREACH(child, &bs->children, next) {

        QDict *new_child_options;

        char *child_key_dot;



        /* reopen can only change the options of block devices that were

         * implicitly created and inherited options. For other (referenced)

         * block devices, a syntax like "backing.foo" results in an error. */

        if (child->bs->inherits_from != bs) {

            continue;

        }



        child_key_dot = g_strdup_printf("%s.", child->name);

        qdict_extract_subqdict(options, &new_child_options, child_key_dot);

        g_free(child_key_dot);



        bdrv_reopen_queue_child(bs_queue, child->bs, new_child_options, 0,

                                child->role, options, flags);

    }



    bs_entry = g_new0(BlockReopenQueueEntry, 1);

    QSIMPLEQ_INSERT_TAIL(bs_queue, bs_entry, entry);



    bs_entry->state.bs = bs;

    bs_entry->state.options = options;

    bs_entry->state.explicit_options = explicit_options;

    bs_entry->state.flags = flags;



    return bs_queue;

}
