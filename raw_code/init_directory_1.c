static int init_directory(BDRVVVFATState* s,const char* dirname)

{

    bootsector_t* bootsector=(bootsector_t*)&(s->first_sectors[(s->first_sectors_number-1)*0x200]);

    unsigned int i;

    unsigned int cluster;



    memset(&(s->first_sectors[0]),0,0x40*0x200);



    /* TODO: if FAT32, this is probably wrong */

    s->sectors_per_fat=0xfc;

    s->sectors_per_cluster=0x10;

    s->cluster_size=s->sectors_per_cluster*0x200;

    s->cluster=malloc(s->cluster_size);

    

    array_init(&(s->mapping),sizeof(mapping_t));

    array_init(&(s->directory),sizeof(direntry_t));

    array_init(&(s->commit),sizeof(commit_t));



    /* add volume label */

    {

	direntry_t* entry=array_get_next(&(s->directory));

	entry->attributes=0x28; /* archive | volume label */

	snprintf(entry->name,11,"QEMU VVFAT");

    }



    if(read_directory(s,dirname,0))

	return -1;



    /* make sure that the number of directory entries is multiple of 0x200/0x20 (to fit the last sector exactly) */

    s->sectors_for_directory=s->directory.next/0x10;



    s->faked_sectors=s->first_sectors_number+s->sectors_per_fat*2+s->sectors_for_directory;

    s->cluster_count=(s->sector_count-s->faked_sectors)/s->sectors_per_cluster;



    /* Now build FAT, and write back information into directory */

    init_fat(s);



    cluster=s->sectors_for_directory/s->sectors_per_cluster;

    assert(s->sectors_for_directory%s->sectors_per_cluster==0);



    /* set the end of the last read directory */

    if(s->first_file_mapping>0) {

	mapping_t* mapping=array_get(&(s->mapping),s->first_file_mapping-1);

	mapping->end=cluster;

    }



    for(i=1;i<s->mapping.next;i++) {

	mapping_t* mapping=array_get(&(s->mapping),i);

	direntry_t* direntry=array_get(&(s->directory),mapping->dir_index);

	if(mapping->mode==MODE_DIRECTORY) {

	    /* directory */

	    int i;

#ifdef DEBUG

	    fprintf(stderr,"assert: %s %d < %d\n",mapping->filename,(int)mapping->begin,(int)mapping->end);

#endif

	    assert(mapping->begin<mapping->end);

	    for(i=mapping->begin;i<mapping->end-1;i++)

		fat_set(s,i,i+1);

	    fat_set(s,i,0x7fffffff);

	} else {

	    /* as the space is virtual, we can be sloppy about it */

	    unsigned int end_cluster=cluster+mapping->end/s->cluster_size;



	    if(end_cluster>=s->cluster_count) {

		fprintf(stderr,"Directory does not fit in FAT%d\n",s->fat_type);

		return -1;

	    }

	    mapping->begin=cluster;

	    mapping->mode=MODE_NORMAL;

	    mapping->offset=0;

	    direntry->size=cpu_to_le32(mapping->end);

	    if(direntry->size==0) {

		direntry->begin=0;

		mapping->end=cluster;

		continue;

	    }



	    direntry->begin=cpu_to_le16(cluster);

	    mapping->end=end_cluster+1;

	    for(;cluster<end_cluster;cluster++)

	        fat_set(s,cluster,cluster+1);

	    fat_set(s,cluster,0x7fffffff);

	    cluster++;

	}

    }



    s->current_mapping=0;



    bootsector->jump[0]=0xeb;

    bootsector->jump[1]=0x3e;

    bootsector->jump[2]=0x90;

    memcpy(bootsector->name,"QEMU    ",8);

    bootsector->sector_size=cpu_to_le16(0x200);

    bootsector->sectors_per_cluster=s->sectors_per_cluster;

    bootsector->reserved_sectors=cpu_to_le16(1);

    bootsector->number_of_fats=0x2; /* number of FATs */

    bootsector->root_entries=cpu_to_le16(s->sectors_of_root_directory*0x10);

    bootsector->zero=0;

    bootsector->media_type=(s->first_sectors_number==1?0xf0:0xf8); /* media descriptor */

    bootsector->sectors_per_fat=cpu_to_le16(s->sectors_per_fat);

    bootsector->sectors_per_track=cpu_to_le16(0x3f);

    bootsector->number_of_heads=cpu_to_le16(0x10);

    bootsector->hidden_sectors=cpu_to_le32(s->first_sectors_number==1?0:0x3f);

    /* TODO: if FAT32, adjust */

    bootsector->total_sectors=cpu_to_le32(s->sector_count);



    /* TODO: if FAT32, this is wrong */

    bootsector->u.fat16.drive_number=0x80; /* assume this is hda (TODO) */

    bootsector->u.fat16.current_head=0;

    bootsector->u.fat16.signature=0x29;

    bootsector->u.fat16.id=cpu_to_le32(0xfabe1afd);



    memcpy(bootsector->u.fat16.volume_label,"QEMU VVFAT ",11);

    memcpy(bootsector->fat_type,(s->fat_type==12?"FAT12   ":s->fat_type==16?"FAT16   ":"FAT32   "),8);

    bootsector->magic[0]=0x55; bootsector->magic[1]=0xaa;



    return 0;

}
