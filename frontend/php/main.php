<?php
//设置需要删除的文件夹
$path = "../core/images/";
//清空文件夹函数和清空文件夹后删除空文件夹函数的处理
function deldir($path){
 //如果是目录则继续
 if(is_dir($path)){
  //扫描一个文件夹内的所有文件夹和文件并返回数组
 $p = scandir($path);
 foreach($p as $val){
  //排除目录中的.和..
  if($val !="." && $val !=".."){
   //如果是目录则递归子目录，继续操作
   if(is_dir($path.$val)){
	//子目录中操作删除文件夹和文件
	deldir($path.$val.'/');
	//目录清空后删除空文件夹
	@rmdir($path.$val.'/');
   }else{
	//如果是文件直接删除
	unlink($path.$val);
   }
  }
 }
}
}
//调用函数，传入路径
deldir($path);

//将用户上传的图片存储到指定位置
if(!file_exists("../core/images/")) //如果文件夹不存在，则创建一个
{
    mkdir("../core/images");  
}
$count=$_POST["count"];
$filesName=$_FILES["pic"]["name"];
$filesTmpNamew = $_FILES['pic']['tmp_name'];  //临时文件名数组
for($i= 0;$i<count($filesName);$i++)  // count():php获取数组长度的方法
{
	if(file_exists('../core/images/'.$filesName[$i])){
        //die($filesName[$i]."文件已存在");  //如果上传的文件已经存在
    }
	else{
		move_uploaded_file($filesTmpNamew[$i], '../core/images/'.$filesName[$i]);  //保存在缓冲区的是临时文件名而不是文件名
	}
}

//调用exe进行计算
$command="F:/web/www/SFM/core/SequentialSfM.exe";
passthru($command);

?>
