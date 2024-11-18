static void vfio_unmap_bar(VFIODevice *vdev, int nr)

{

    VFIOBAR *bar = &vdev->bars[nr];



    if (!bar->size) {

        return;

    }



    vfio_bar_quirk_teardown(vdev, nr);



    memory_region_del_subregion(&bar->mem, &bar->mmap_mem);

    munmap(bar->mmap, memory_region_size(&bar->mmap_mem));




    if (vdev->msix && vdev->msix->table_bar == nr) {

        memory_region_del_subregion(&bar->mem, &vdev->msix->mmap_mem);

        munmap(vdev->msix->mmap, memory_region_size(&vdev->msix->mmap_mem));


    }



    memory_region_destroy(&bar->mem);

}