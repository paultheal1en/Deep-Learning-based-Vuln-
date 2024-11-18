static int vpc_create(const char *filename, QEMUOptionParameter *options)

{

    uint8_t buf[1024];

    struct vhd_footer* footer = (struct vhd_footer*) buf;

    struct vhd_dyndisk_header* dyndisk_header =

        (struct vhd_dyndisk_header*) buf;

    int fd, i;

    uint16_t cyls;

    uint8_t heads;

    uint8_t secs_per_cyl;

    size_t block_size, num_bat_entries;

    int64_t total_sectors = 0;



    // Read out options

    while (options && options->name) {

        if (!strcmp(options->name, "size")) {

            total_sectors = options->value.n / 512;

        }

        options++;

    }



    // Create the file

    fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0644);

    if (fd < 0)

        return -EIO;



    // Calculate matching total_size and geometry

    if (calculate_geometry(total_sectors, &cyls, &heads, &secs_per_cyl))

        return -EFBIG;

    total_sectors = (int64_t) cyls * heads * secs_per_cyl;



    // Prepare the Hard Disk Footer

    memset(buf, 0, 1024);



    memcpy(footer->creator, "conectix", 8);

    // TODO Check if "qemu" creator_app is ok for VPC

    memcpy(footer->creator_app, "qemu", 4);

    memcpy(footer->creator_os, "Wi2k", 4);



    footer->features = be32_to_cpu(0x02);

    footer->version = be32_to_cpu(0x00010000);

    footer->data_offset = be64_to_cpu(HEADER_SIZE);

    footer->timestamp = be32_to_cpu(time(NULL) - VHD_TIMESTAMP_BASE);



    // Version of Virtual PC 2007

    footer->major = be16_to_cpu(0x0005);

    footer->minor =be16_to_cpu(0x0003);



    footer->orig_size = be64_to_cpu(total_sectors * 512);

    footer->size = be64_to_cpu(total_sectors * 512);



    footer->cyls = be16_to_cpu(cyls);

    footer->heads = heads;

    footer->secs_per_cyl = secs_per_cyl;



    footer->type = be32_to_cpu(VHD_DYNAMIC);



    // TODO uuid is missing



    footer->checksum = be32_to_cpu(vpc_checksum(buf, HEADER_SIZE));



    // Write the footer (twice: at the beginning and at the end)

    block_size = 0x200000;

    num_bat_entries = (total_sectors + block_size / 512) / (block_size / 512);



    if (write(fd, buf, HEADER_SIZE) != HEADER_SIZE)

        return -EIO;



    if (lseek(fd, 1536 + ((num_bat_entries * 4 + 511) & ~511), SEEK_SET) < 0)

        return -EIO;

    if (write(fd, buf, HEADER_SIZE) != HEADER_SIZE)

        return -EIO;



    // Write the initial BAT

    if (lseek(fd, 3 * 512, SEEK_SET) < 0)

        return -EIO;



    memset(buf, 0xFF, 512);

    for (i = 0; i < (num_bat_entries * 4 + 511) / 512; i++)

        if (write(fd, buf, 512) != 512)

            return -EIO;





    // Prepare the Dynamic Disk Header

    memset(buf, 0, 1024);



    memcpy(dyndisk_header->magic, "cxsparse", 8);



    dyndisk_header->data_offset = be64_to_cpu(0xFFFFFFFF);

    dyndisk_header->table_offset = be64_to_cpu(3 * 512);

    dyndisk_header->version = be32_to_cpu(0x00010000);

    dyndisk_header->block_size = be32_to_cpu(block_size);

    dyndisk_header->max_table_entries = be32_to_cpu(num_bat_entries);



    dyndisk_header->checksum = be32_to_cpu(vpc_checksum(buf, 1024));



    // Write the header

    if (lseek(fd, 512, SEEK_SET) < 0)

        return -EIO;

    if (write(fd, buf, 1024) != 1024)

        return -EIO;



    close(fd);

    return 0;

}
