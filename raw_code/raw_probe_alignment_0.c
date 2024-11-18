static void raw_probe_alignment(BlockDriverState *bs)

{

    BDRVRawState *s = bs->opaque;

    DWORD sectorsPerCluster, freeClusters, totalClusters, count;

    DISK_GEOMETRY_EX dg;

    BOOL status;



    if (s->type == FTYPE_CD) {

        bs->request_alignment = 2048;

        return;

    }

    if (s->type == FTYPE_HARDDISK) {

        status = DeviceIoControl(s->hfile, IOCTL_DISK_GET_DRIVE_GEOMETRY_EX,

                                 NULL, 0, &dg, sizeof(dg), &count, NULL);

        if (status != 0) {

            bs->request_alignment = dg.Geometry.BytesPerSector;

            return;

        }

        /* try GetDiskFreeSpace too */

    }



    if (s->drive_path[0]) {

        GetDiskFreeSpace(s->drive_path, &sectorsPerCluster,

                         &dg.Geometry.BytesPerSector,

                         &freeClusters, &totalClusters);

        bs->request_alignment = dg.Geometry.BytesPerSector;

    }

}
