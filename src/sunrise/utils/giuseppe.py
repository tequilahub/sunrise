import sys

def giuseppe():
    giuseppe = r'''
                                                        .
            ___---""""""""""""""---____                / |
       _--""   \)))))))))))))))))))))))"""""___       /  |
    _-" _       \))))))_-"|))))))))))))))))))))"""---' __|
 _-"   / \       |))))|   |)))))))))))))))))))))))))/--  |
&lt;___.  \_/       |))))|   |))))))))))))))))))))))))&lt;-    |
 "-_             |))))|   |)))))))))))))))))))))))))\--__|
    "-_         /)))))) -_|))))))))))))))))))))___---.   |
       "--__   /)))))))))))))))))))))))_____"""       \  |
            """---______________---""""        mga     \ |
                                                        '
    '''
    print(giuseppe)

def giuseppe_bar(step:int,total_steps:int,toolbar_width:int = 50):
    toolbar_width = total_steps*(toolbar_width//total_steps)
    step_width = toolbar_width//total_steps
    i = step*step_width
    sys.stdout.write(f"\r\b[{i*'~'}><(((º>{(toolbar_width-i)*' '}]")
    sys.stdout.flush()
