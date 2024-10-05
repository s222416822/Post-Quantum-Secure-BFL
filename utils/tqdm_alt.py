try:
    # Try to import tqdm for progress bar functionality
    from tqdm import tqdm

except ImportError:
    # Handle missing tqdm module
    import sys
    from time import time

    # Print warning about missing tqdm and provide installation instructions
    print('Missing tqdm module, alternative logging system activated', file=sys.stderr)
    print('You can install tqdm with `python3 -m pip install tqdm`', file=sys.stderr)


    # Define a basic alternative for tqdm
    def tqdm(iterable, total=None):
        # If total isn't provided, calculate it from the iterable
        if total is None:
            total = len(iterable)

        # Record start time
        dbt = time()

        # Loop through the iterable with progress tracking
        for i, line in zip(range(total), iterable):
            yield line
            # Update progress percentage and print to stderr
            if (i - 1) * 100 // total < i * 100 // total:
                print('\r%u%%' % (i * 100 // total), end='', file=sys.stderr)
        print('\r100%', file=sys.stderr)

        # Print total time taken
        print('Finished in %u seconds' % (time() - dbt), file=sys.stderr)
