BlockStatsList *qmp_query_blockstats(Error **errp)
{
    BlockStatsList *head = NULL, **p_next = &head;
    BlockDriverState *bs = NULL;
     while ((bs = bdrv_next(bs))) {
        BlockStatsList *info = g_malloc0(sizeof(*info));
        info->value = bdrv_query_stats(bs);
        *p_next = info;
        p_next = &info->next;
    }
    return head;
}