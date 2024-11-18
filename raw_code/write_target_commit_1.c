static int write_target_commit(BlockDriverState *bs, int64_t sector_num,

	const uint8_t* buffer, int nb_sectors) {

    BDRVVVFATState* s = bs->opaque;

    return try_commit(s);

}
