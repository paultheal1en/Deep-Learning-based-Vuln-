static struct mm_struct *vma_init(void)

{

    struct mm_struct *mm;



    if ((mm = qemu_malloc(sizeof (*mm))) == NULL)

        return (NULL);



    mm->mm_count = 0;

    TAILQ_INIT(&mm->mm_mmap);



    return (mm);

}
