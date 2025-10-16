#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查diffusion_policy/config/Pre-Train目录下的YAML文件，
删除文件名中包含'backup'关键字的文件。
"""

import os
import argparse
import shutil
from pathlib import Path
import yaml


def is_yaml_file(file_path):
    """检查文件是否为YAML文件"""
    return file_path.suffix.lower() in ['.yaml', '.yml']


def contains_backup(file_name):
    """检查文件名是否包含'backup'关键字（不区分大小写）"""
    return 'backup' in file_name.lower()


def get_file_info(file_path):
    """获取文件详细信息"""
    stat = file_path.stat()
    return {
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'modified_time': stat.st_mtime,
        'created_time': stat.st_ctime if hasattr(stat, 'st_ctime') else None
    }


def move_to_trash(file_path, trash_dir):
    """将文件移动到回收站目录（更安全的删除方式）"""
    try:
        # 创建回收站目录
        trash_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名，避免冲突
        original_name = file_path.name
        timestamp = int(file_path.stat().st_mtime)
        trash_name = f"{original_name}_deleted_{timestamp}"
        trash_path = trash_dir / trash_name
        
        # 移动文件到回收站
        file_path.rename(trash_path)
        return True, trash_path
    except Exception as e:
        print(f"  移动到回收站失败: {e}")
        return False, None


def delete_file_safely(file_path, trash_dir, dry_run=False):
    """安全删除文件，支持回收站机制和预览模式"""
    if dry_run:
        return True, "预览模式：跳过实际删除"
    
    # 优先尝试移动到回收站
    success, result_path = move_to_trash(file_path, trash_dir)
    if success:
        return True, f"已移动到回收站: {result_path.name}"
    
    # 如果移动失败，直接删除
    try:
        file_path.unlink()
        return True, "已直接删除"
    except Exception as e:
        return False, f"删除失败: {e}"


def process_backup_files(pretrain_dir, trash_dir=None, dry_run=False):
    """处理包含backup关键字的YAML文件"""
    pretrain_path = Path(pretrain_dir)
    
    if not pretrain_path.exists():
        print(f"错误: 目录 {pretrain_path} 不存在")
        return
    
    if not pretrain_path.is_dir():
        print(f"错误: {pretrain_path} 不是一个目录")
        return
    
    # 确定回收站目录
    if trash_dir is None:
        trash_dir = pretrain_path / ".trash"
    
    # 查找所有YAML文件
    yaml_files = []
    for file_path in pretrain_path.rglob("*.yaml"):
        if is_yaml_file(file_path):
            yaml_files.append(file_path)
    for file_path in pretrain_path.rglob("*.yml"):
        if is_yaml_file(file_path):
            yaml_files.append(file_path)
    
    if not yaml_files:
        print(f"在 {pretrain_path} 中未找到YAML文件")
        return
    
    # 筛选包含backup的文件
    backup_files = [f for f in yaml_files if contains_backup(f.name)]
    
    if not backup_files:
        print(f"在 {pretrain_path} 中未找到包含'backup'关键字的YAML文件")
        return
    
    print(f"找到 {len(backup_files)} 个包含'backup'的YAML文件:")
    
    processed_count = 0
    success_count = 0
    total_size = 0
    
    for file_path in backup_files:
        processed_count += 1
        file_info = get_file_info(file_path)
        total_size += file_info['size_mb']
        
        print(f"\n[{processed_count}/{len(backup_files)}] {file_path.name}")
        print(f"  大小: {file_info['size_mb']} MB")
        
        # 执行删除操作
        success, message = delete_file_safely(file_path, trash_dir, dry_run)
        if success:
            print(f"  ✓ {message}")
            success_count += 1
        else:
            print(f"  ✗ {message}")
    
    print(f"\n{'='*60}")
    print(f"处理完成:")
    print(f"  总计处理文件: {processed_count}")
    print(f"  成功删除/移动: {success_count}")
    print(f"  释放空间约: {round(total_size, 2)} MB")
    
    if dry_run:
        print(f"\n注意: 这是预览模式，未执行任何删除操作")
        print(f"回收站目录: {trash_dir}")
        print("使用 --dry-run 标志移除预览模式以实际执行删除")
    else:
        print(f"\n回收站位置: {trash_dir}")
        print("如需彻底删除，请手动清理回收站目录")
        print("如需恢复文件，可从回收站目录中找回")


def main():
    parser = argparse.ArgumentParser(description="删除包含'backup'关键字的YAML文件")
    parser.add_argument(
        "--pretrain-dir",
        default="diffusion_policy/config/Pre-Train",
        help="要检查的Pre-Train目录路径 (默认: diffusion_policy/config/Pre-Train)"
    )
    parser.add_argument(
        "--trash-dir",
        help="自定义回收站目录路径 (默认: 在原目录下创建 .trash 文件夹)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式：显示将要删除的文件，但不实际执行删除"
    )
    parser.add_argument(
        "--force-delete",
        action="store_true",
        help="强制直接删除文件，不移动到回收站（谨慎使用）"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("=== 预览模式（Dry Run）===")
        print("以下文件将被删除/移动到回收站，但不会实际执行操作:")
        print("-" * 50)
    
    # 如果使用强制删除，则不使用回收站
    if args.force_delete:
        print("警告: 使用强制删除模式，文件将被永久删除!")
        print("此操作不可逆，请确认后再执行")
        trash_dir = None
    else:
        trash_dir = Path(args.trash_dir) if args.trash_dir else None
    
    process_backup_files(
        args.pretrain_dir, 
        trash_dir=trash_dir, 
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()