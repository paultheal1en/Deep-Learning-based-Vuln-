static inline int mirror_clip_sectors(MirrorBlockJob *s,

                                      int64_t sector_num,

                                      int nb_sectors)

{

    return MIN(nb_sectors,

               s->bdev_length / BDRV_SECTOR_SIZE - sector_num);

}
