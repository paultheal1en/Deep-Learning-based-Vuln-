ram_addr_t ppc4xx_sdram_adjust(ram_addr_t ram_size, int nr_banks,

                               target_phys_addr_t ram_bases[],

                               target_phys_addr_t ram_sizes[],

                               const unsigned int sdram_bank_sizes[])

{

    ram_addr_t size_left = ram_size;

    int i;

    int j;



    for (i = 0; i < nr_banks; i++) {

        for (j = 0; sdram_bank_sizes[j] != 0; j++) {

            unsigned int bank_size = sdram_bank_sizes[j];



            if (bank_size <= size_left) {

                char name[32];

                snprintf(name, sizeof(name), "ppc4xx.sdram%d", i);

                ram_bases[i] = qemu_ram_alloc(NULL, name, bank_size);

                ram_sizes[i] = bank_size;

                size_left -= bank_size;

                break;

            }

        }



        if (!size_left) {

            /* No need to use the remaining banks. */

            break;

        }

    }



    ram_size -= size_left;

    if (size_left)

        printf("Truncating memory to %d MiB to fit SDRAM controller limits.\n",

               (int)(ram_size >> 20));



    return ram_size;

}
