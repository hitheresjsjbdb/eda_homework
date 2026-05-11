
"""Here are utilities for daily regression."""

import subprocess
import logging
import sys
import re
import os
import time

def config_logger(logfile):
    logger = logging.getLogger('root')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('[%(asctime)s]%(filename)s:%(lineno)d:[%(levelname)s]: %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(lambda x: x.levelno <= logging.INFO)
    stdout_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(fmt)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger

def run_subprocess(arg):
    """ A wrapper of subprocess.Popen """
    cmd, logger = arg
    p = subprocess.Popen(cmd, shell=True, 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout, stderr = stdout.decode(), stderr.decode()
    if logger:
        if stderr:
            logger.error(stderr)
            logger.error(f'Failed to run cmd: {cmd}')
        else:
            logger.debug(stdout)
            logger.info(f'Finished to run cmd: {cmd}')
    else:
        print(stdout)
        print(stderr)
    return p.returncode

class Subprocess(object):
    # A wrapper of subprocess.Popen
    @staticmethod
    def run(cmd, timeout, logger):
        proc = subprocess.Popen(cmd, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        logger.debug('start to run cmd "{}" '.format(cmd))
        try:
            start_time = time.time()
            stdout, stderr = proc.communicate(timeout=timeout)
            stdout, stderr = stdout.decode(), stderr.decode()
            runtime = f'{time.time() - start_time:.3f}'
        except subprocess.TimeoutExpired:
            # kill subprocess with it's desendants
            ptree = subprocess.Popen(f'pstree -p {proc.pid}', shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE).communicate()[0].decode()
            for m in re.finditer(r'\d+', ptree):
                cpid = int(m.group())
                try:
                    os.kill(cpid, 9)
                except:
                    pass
            logger.error('Failed to run cmd "{}" with timeout'.format(cmd))
            stdout, stderr, runtime = '', 'timeout', timeout 

        if stderr:
            #logger.error('Failed to run cmd "{}"'.format(cmd))
            logger.critical(stderr)
        else:
            logger.debug(stdout)

        logger.debug('finished to run cmd "{}"'.format(cmd))
        return stdout, stderr, runtime
