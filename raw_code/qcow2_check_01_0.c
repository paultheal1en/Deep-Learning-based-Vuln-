static int qcow2_check(BlockDriverState *bs, BdrvCheckResult *result)

{

    return qcow2_check_refcounts(bs, result);

}
