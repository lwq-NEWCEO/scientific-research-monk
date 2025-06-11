import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from '@/components/ui/table';

const TeacherQueryPage = () => {
  // --- 日志：组件渲染 ---
  console.log('[TeacherQueryPage] Component rendering...');

   // --- 原始静态数据 ---
  const [allTeachers] = useState([
    // --- 上海交通大学 ---
    { id: 1, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '陈海波', tag: '重科研', email: 'haibochen@sjtu.edu.cn', status: '已上线' },
    { id: 2, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '蔡鸿明', tag: '重科研', email: 'hmcai@sjtu.edu.', status: '已上线' },
    { id: 3, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '陈榕', tag: '重科研', email: 'rongchen@sjtu.eud.cn', status: '已上线' },
    { id: 4, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '董明凯', tag: '重科研', email: 'mingkaidong@sjtu.edu.cn', status: '已上线' },
    { id: 5, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '夏虞斌', tag: '重科研', email: 'xiayubin@sjtu.edu.cn', status: '已上线' },
    { id: 6, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '臧斌宇', tag: '重科研', email: 'byzang@sjtu.edu.cn', status: '已上线' },
    { id: 7, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '杜东', tag: '重科研', email: 'Dd_nirvana@sjtu.edu.cn', status: '已上线' },
    { id: 8, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '王肇国', tag: '重科研', email: 'zhaoguowang@sjtu.edu.cn', status: '已上线' },
    { id: 9, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '吴明瑜', tag: '重科研', email: '--', status: '已上线' },
    { id: 10, university: '上海交通大学', researchDirection: '计算机系统', department: '软件学院', teacher: '华志超', tag: '重科研', email: 'zchua@sjtu.edu.cn', status: '已上线' },

    // --- 复旦大学 (已有数据) ---
    { id: 11, university: '复旦大学', researchDirection: '人工智能', department: '计算机学院', teacher: '李老师', tag: '优青', email: 'li@fudan.edu.cn', status: '已上线' },

    // --- 复旦大学 (新增数据) ---
    { id: 12, university: '复旦大学', researchDirection: '嵌入式系统', department: '计算机学院', teacher: '陈伟男', tag: '论文多', email: 'wnchen@fudan.edu.cn', status: '已上线' },
    { id: 13, university: '复旦大学', researchDirection: '智能互联网', department: '计算机学院', teacher: '陈阳', tag: '项目多', email: 'chenyang@fudan.edu.cn', status: '已上线' },
    { id: 14, university: '复旦大学', researchDirection: '计算机视觉', department: '计算机学院', teacher: '戈维峰', tag: '开源项目多', email: 'wfge@fudan.edu.cn', status: '已上线' },
    { id: 15, university: '复旦大学', researchDirection: '编码学', department: '计算机学院', teacher: '金玲飞', tag: '经验丰富', email: 'lfjin@fudan.edu.cn', status: '已上线' },

    // --- 你可以在这里继续添加更多学校或老师的数据 ---

  ]);


  // --- 表单筛选条件的状态 ---
  const [university, setUniversity] = useState('');
  const [researchDirection, setResearchDirection] = useState('');
  const [department, setDepartment] = useState('');
  const [tag, setTag] = useState('');
  const [title, setTitle] = useState('全部');

  // --- 用于表格展示的过滤后教师列表状态 ---
  const [filteredTeachers, setFilteredTeachers] = useState(allTeachers);

  // --- [新] 包装状态更新函数以添加日志 ---
  const updateUniversity = (value) => {
    console.log(`[Update] Setting university to: ${value}`);
    setUniversity(value);
  };
  const updateResearchDirection = (event) => {
    const value = event.target.value;
    console.log(`[Update] Setting researchDirection to: ${value}`);
    setResearchDirection(value);
  };
   const updateDepartment = (value) => {
    console.log(`[Update] Setting department to: ${value}`);
    setDepartment(value);
  };
   const updateTag = (value) => {
    console.log(`[Update] Setting tag to: ${value}`);
    setTag(value);
  };
  const updateTitle = (value) => {
    console.log(`[Update] Setting title to: ${value}`);
    setTitle(value);
  };

  // --- 使用 useEffect 实现实时筛选 ---
  useEffect(() => {
    // --- 日志：useEffect 触发 ---
    console.log('[useEffect] Running filter due to dependency change...');
    console.log('[useEffect] Current filter state:', { university, researchDirection, department, tag, title });

    const filtered = allTeachers.filter(teacher => {
      const universityMatch = university === '' || teacher.university.includes(university);
      const researchDirectionMatch = researchDirection === '' || teacher.researchDirection.toLowerCase().includes(researchDirection.toLowerCase());
      const departmentMatch = department === '' || teacher.department.includes(department);
      const tagMatch = tag === '' || teacher.tag.includes(tag);
      const titleMatch = title === '全部' || true;

      return universityMatch && researchDirectionMatch && departmentMatch && tagMatch && titleMatch;
    });

    // --- 日志：筛选结果 ---
    console.log('[useEffect] Filtering complete. Filtered count:', filtered.length);
    setFilteredTeachers(filtered);

  }, [university, researchDirection, department, tag, title, allTeachers]); // 依赖项数组

  // --- 重置筛选条件 ---
  const handleReset = () => {
    // --- 日志：重置触发 ---
    console.log('[handleReset] Resetting filters...');
    updateUniversity(''); // 使用包装函数
    // 注意：Input 的 reset 需要直接设置状态，因为它是由 event 驱动的
    console.log(`[Update] Setting researchDirection to: ''`);
    setResearchDirection('');
    updateDepartment('');
    updateTag('');
    updateTitle('全部');
  };

  // --- 分页状态 (如果需要) ---
  const [currentPage] = useState(1);
  const totalItems = filteredTeachers.length;

  // --- JSX 渲染部分 (使用包装后的更新函数) ---
  return (
    <div className="p-4 md:p-6 space-y-4 bg-gray-50 min-h-screen">
        {/* ... (Breadcrumb, Title) ... */}
        <div className="text-sm text-gray-500 mb-4">/ 列表页 / 导师查询</div>
        <h1 className="text-xl font-semibold mb-4">老师查询</h1>

        {/* Search Form Card */}
        <Card>
            <CardContent className="p-4 space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 items-end">
                    {/* 所属大学 */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">所属大学</label>
                        {/* 使用包装后的 updateUniversity */}
                        <Select onValueChange={updateUniversity} value={university}>
                            <SelectTrigger><SelectValue placeholder="选择大学" /></SelectTrigger>
                            <SelectContent>
                                <SelectItem value="上海交通大学">上海交通大学</SelectItem>
                                <SelectItem value="复旦大学">复旦大学</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    {/* 研究方向 */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">研究方向</label>
                        {/* 使用包装后的 updateResearchDirection */}
                        <Input placeholder="输入研究方向" value={researchDirection} onChange={updateResearchDirection} />
                    </div>
                    {/* 院系 */}
                     <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">院系</label>
                        <Select onValueChange={updateDepartment} value={department}>
                           <SelectTrigger><SelectValue placeholder="选择院系" /></SelectTrigger>
                            <SelectContent>
                                <SelectItem value="软件学院">软件学院</SelectItem>
                                <SelectItem value="计算机学院">计算机学院</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    {/* 标签 */}
                     <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">标签</label>
                        <Select onValueChange={updateTag} value={tag}>
                            <SelectTrigger><SelectValue placeholder="选择标签" /></SelectTrigger>
                            <SelectContent>
                                <SelectItem value="重科研">重科研</SelectItem>
                                <SelectItem value="优青">优青</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    {/* 职称 */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">职称</label>
                         <Select onValueChange={updateTitle} value={title}>
                            <SelectTrigger><SelectValue placeholder="选择职称" /></SelectTrigger>
                            <SelectContent>
                                <SelectItem value="全部">全部</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    {/* 重置按钮 */}
                     <div className="flex justify-end space-x-2 lg:col-start-5">
                        <Button variant="outline" onClick={handleReset}>重置</Button>
                    </div>
                </div>
            </CardContent>
        </Card>

        {/* Action Buttons & Table Card */}
        <Card>
            <CardContent className="p-4">
                 {/* ... (表格上方按钮) ... */}
                 <div className="flex justify-between items-center mb-4">
                    <div className="space-x-2">
                         <Button>+ 新建</Button>
                         <Button variant="outline">批量导入</Button>
                    </div>
                    <Button variant="outline">下载</Button>
                </div>

                {/* 表格 */}
                <Table>
                    {/* ... (TableHeader) ... */}
                     <TableHeader>
                        <TableRow>
                            <TableHead>所属大学</TableHead>
                            <TableHead>研究方向</TableHead>
                            <TableHead>院系</TableHead>
                            <TableHead>导师</TableHead>
                            <TableHead>标签</TableHead>
                            <TableHead>邮箱</TableHead>
                            <TableHead>内容情况</TableHead>
                            <TableHead>操作</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {/* --- 日志：渲染表格行数 --- */}
                        {console.log('[TableBody] Rendering table with', filteredTeachers.length, 'rows.')}
                        {filteredTeachers.length > 0 ? (
                            filteredTeachers.map((teacher) => (
                                <TableRow key={teacher.id}>
                                    {/* ... (TableCell) ... */}
                                    <TableCell>{teacher.university}</TableCell>
                                    <TableCell>{teacher.researchDirection}</TableCell>
                                    <TableCell>{teacher.department}</TableCell>
                                    <TableCell>{teacher.teacher}</TableCell>
                                    <TableCell>{teacher.tag}</TableCell>
                                    <TableCell>{teacher.email}</TableCell>
                                    <TableCell>{teacher.status}</TableCell>
                                    <TableCell><Button variant="link" className="p-0 h-auto text-blue-600">查看</Button></TableCell>
                                </TableRow>
                            ))
                        ) : (
                            <TableRow>
                                <TableCell colSpan={8} className="text-center py-4">暂无符合条件的教师数据</TableCell>
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
                 {/* ... (分页) ... */}
                 <div className="flex justify-end items-center mt-4 text-sm text-gray-600">
                     <span>共 {totalItems} 条</span>
                     <div className="flex items-center space-x-1 ml-4">
                         <Button variant="outline" size="icon" className="h-8 w-8" disabled={currentPage === 1}>&lt;</Button>
                         <Button variant={currentPage === 1 ? "default" : "outline"} size="icon" className="h-8 w-8">1</Button>
                         <Button variant="outline" size="icon" className="h-8 w-8">&gt;</Button>
                     </div>
                 </div>
            </CardContent>
        </Card>
    </div>
  );
};

export default TeacherQueryPage;
