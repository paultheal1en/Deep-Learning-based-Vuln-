static inline bool check_lba_range(SCSIDiskState *s,

                                   uint64_t sector_num, uint32_t nb_sectors)

{

    /*

     * The first line tests that no overflow happens when computing the last

     * sector.  The second line tests that the last accessed sector is in

     * range.

     */

    return (sector_num <= sector_num + nb_sectors &&

            sector_num + nb_sectors - 1 <= s->qdev.max_lba);

}
